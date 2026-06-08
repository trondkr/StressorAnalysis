"""
Dual-Baseline Marine Stressor Analysis
=======================================
Companion module to advanced_stressor_metrics.py. Adds:

  1. Dual-baseline detection (fixed + shifting) following Hobday et al. (2016)
     and the shifting-baseline recommendation of Amaya et al. (2023).
  2. Chronicity transition diagnostics: identifies when each stressor crosses
     from episodic to chronic under each baseline definition.
  3. Cascade panel **(c)** (Washington default): **CascadeFidelity.lift** — ratio
     P(response|recent driver)/P(response) on the **same 1D** monthly flags as
     panels (a–b). Use ``--panel-c gridded`` for block-averaged gridded
     ``cascading_*`` % (``AdvancedStressorMetrics``, as in
     ``dual_baseline_sequential_cascade_stats.py``).
  4. Publication-quality figure generators that match the ones produced for
     the improved Figure 1 and the proposed new multi-stressor analysis.

Scientific rationale
--------------------
Fixed-baseline percentile detection conflates three distinct signals under
climate change:
    (i)   long-term warming (trend)
    (ii)  discrete extreme events (true MHWs/hypoxia events)
    (iii) interaction between them
By mid-century under SSP2-4.5 projections, the time series spends the majority
of its time above the historical p90 threshold, so "events" fuse into
multi-year exceedances — the perpetual-heatwave artifact.

The shifting baseline (equivalent to adding the linear warming trend to the
fixed threshold, or detrending the data before detection) keeps the three
signals separated. Each retains interpretive value:
    - Fixed baseline  => "total heat exposure" (Amaya et al. 2023 terminology)
    - Shifting baseline => "marine heatwave" sensu stricto

Usage
-----
CLI (Washington Quinault ROMS via ``ACTEA_gcs``, default)::

    PYTHONPATH=/path/to/ACTEA_downscale python StressorAnalysisV2/dual_baseline_stressor_analysis.py \\
        --scenario ssp245 --output-dir ./outputs --project WA_state

    PNGs: ``{project}_heatwaves_fixed_shifting_baseline_{scenario}.png`` and
    ``{project}_chronicity_cascade_{scenario}.png``.  Washington defaults use the
    **full ocean grid mean** (``monthly_domain_mean`` / ``sst_daily_domain``);
    pass ``--spatial-domain coastal`` for the old coastal (<100 m) series.
    For gridded chronic-emergence / cascade maps, use ``dual_baseline_spatial_emergence.py``.

Synthetic demo::

    python dual_baseline_stressor_analysis.py --demo -o ./outputs

Library::

    from dual_baseline_stressor_analysis import DualBaselineDetector, \
        ChronicityMetrics, CascadeFidelity, plot_improved_figure1, \
        plot_chronicity_and_cascade

    # 1D time series (pandas Series indexed by datetime, or xr.DataArray with time dim)
    detector = DualBaselineDetector(
        baseline_years=(2017, 2022),
        percentile=90,
        min_duration=5,           # days; use 3 for monthly data with unit='month'
        direction='above',        # 'above' for MHWs; 'below' for hypoxia/OA
    )
    result = detector.detect(sst_series)
    # result = {'fixed': {...}, 'shifting': {...}, 'metadata': {...}}

    # Annual exposure %
    exp_fixed = ChronicityMetrics.annual_exposure(result['fixed']['flag'])
    exp_shift = ChronicityMetrics.annual_exposure(result['shifting']['flag'])

    # Chronic transition year (first 3 consecutive yrs >=50% exposure)
    yr_fixed = ChronicityMetrics.chronic_transition_year(exp_fixed, threshold=50, run=3)

    # Cascade fidelity
    lift_df = CascadeFidelity.lift(bhw_flag, hyp_flag, window_days=30, block_years=5)

    # Figures
    fig1 = plot_improved_figure1(result, region_name='Washington coast')
    fig2 = plot_chronicity_and_cascade(stressor_results, cascade_pairs)

Gridded (xr.DataArray / xr.Dataset) workflow
--------------------------------------------
For spatially-resolved detection, use `DualBaselineDetector.detect_gridded`,
which applies the 1D detector along the time axis of each grid cell via
xarray.apply_ufunc (Dask-compatible).

Author: adapted for QIN v6 manuscript
Date: April 2026
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ============================================================================
# MAP STYLING (used by dual_baseline_spatial_emergence; year colormap helpers)
# ============================================================================

# Eastern longitude bound (Plate Carree): crop inland / eastern land in viewers.
DUAL_BASELINE_MAP_LON_EAST_CLIP: float = -123.3


def dual_baseline_truncated_twilight(
    n: int, hi: float = 0.9,
) -> 'matplotlib.colors.ListedColormap':
    """Sample cyclic ``twilight`` on ``[0, hi]`` so the top bin does not repeat the bottom hue."""
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    base = plt.get_cmap('twilight')
    rgba = base(np.linspace(0.0, hi, int(n), endpoint=True))
    return mcolors.ListedColormap(rgba)


def dual_baseline_year_boundary_cmap_norm(
    vmin: int, vmax: int, n_bins: int = 9,
) -> Tuple['matplotlib.colors.ListedColormap', 'matplotlib.colors.BoundaryNorm']:
    """Discrete year maps: truncated twilight + ``BoundaryNorm``."""
    import matplotlib.colors as mcolors

    cmap = dual_baseline_truncated_twilight(n_bins, hi=0.9)
    bounds = np.linspace(vmin, vmax, n_bins + 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N, clip=False)
    return cmap, norm


def dual_baseline_map_extent_lon_lat(
    lon: np.ndarray,
    lat: np.ndarray,
    lon_east_clip: float = DUAL_BASELINE_MAP_LON_EAST_CLIP,
) -> Tuple[float, float, float, float]:
    """``(lon_west, lon_east, lat_south, lat_north)`` with eastern view clipped to ``lon_east_clip``."""
    w = float(np.min(lon))
    e_dat = float(np.max(lon))
    e = min(e_dat, lon_east_clip)
    if e <= w:
        e = e_dat
    s = float(np.min(lat))
    n = float(np.max(lat))
    return w, e, s, n


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class DetectionResult:
    """Container for the output of a single detection (one baseline, one series).

    Attributes
    ----------
    events : pd.DataFrame
        One row per event with start, end, duration, intensity, category columns.
    flag : pd.Series
        Boolean time series indicating in-event timesteps.
    threshold : pd.Series
        Time-varying threshold used for detection (day-of-year for fixed,
        day-of-year + trend for shifting).
    climatology : pd.Series
        Day-of-year climatological mean used as reference.
    baseline : str
        Either 'fixed' or 'shifting'.
    """
    events: pd.DataFrame
    flag: pd.Series
    threshold: pd.Series
    climatology: pd.Series
    baseline: str

    def annual_exposure(self) -> pd.Series:
        return ChronicityMetrics.annual_exposure(self.flag)


@dataclass
class DualResult:
    """Combined fixed and shifting detection for one variable."""
    fixed: DetectionResult
    shifting: DetectionResult
    variable: str
    trend_per_year: float
    baseline_years: Tuple[int, int]


# ============================================================================
# LOW-LEVEL 1D DETECTION FUNCTIONS
# ============================================================================

def _pad_smooth(x: np.ndarray, window: int) -> np.ndarray:
    """Circular-padded moving average suitable for day-of-year climatologies."""
    if window <= 1:
        return x.copy()
    k = np.ones(window) / window
    xp = np.concatenate([x[-window:], x, x[:window]])
    return np.convolve(xp, k, mode='same')[window:-window]


def _doy_climatology(
    values: np.ndarray,
    doy: np.ndarray,
    baseline_mask: np.ndarray,
    percentile: float,
    half_window_days: int = 5,
    smooth_window: int = 31,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build day-of-year climatology and percentile threshold (Hobday 2016 style).

    Pools ±half_window_days around each DOY across all baseline years,
    then applies a `smooth_window`-day circular moving average.

    Returns
    -------
    clim : np.ndarray, length 366
    thresh : np.ndarray, length 366
    """
    clim = np.full(366, np.nan)
    thresh = np.full(366, np.nan)
    q = percentile / 100.0

    for d in range(1, 367):
        pool_idx: List[int] = []
        for k in range(-half_window_days, half_window_days + 1):
            target = ((d - 1 + k) % 366) + 1
            pool_idx.extend(np.where((doy == target) & baseline_mask)[0].tolist())
        if pool_idx:
            vals = values[pool_idx]
            vals = vals[~np.isnan(vals)]
            if len(vals):
                clim[d - 1] = vals.mean()
                thresh[d - 1] = np.quantile(vals, q)

    # Fill any NaN DOY with nearest valid
    def _fill_nan(x: np.ndarray) -> np.ndarray:
        if np.any(np.isnan(x)):
            good = ~np.isnan(x)
            if good.any():
                x = np.interp(np.arange(366), np.where(good)[0], x[good])
        return x

    clim = _fill_nan(clim)
    thresh = _fill_nan(thresh)

    clim = _pad_smooth(clim, smooth_window)
    thresh = _pad_smooth(thresh, smooth_window)
    return clim, thresh


def _monthly_climatology(
    values: np.ndarray,
    month: np.ndarray,
    baseline_mask: np.ndarray,
    percentile: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Monthly (1..12) climatology for coarser-resolution data."""
    clim = np.full(12, np.nan)
    thresh = np.full(12, np.nan)
    q = percentile / 100.0
    for m in range(1, 13):
        sel = (month == m) & baseline_mask
        if sel.any():
            vals = values[sel]
            vals = vals[~np.isnan(vals)]
            if len(vals):
                clim[m - 1] = vals.mean()
                thresh[m - 1] = np.quantile(vals, q)
    return clim, thresh


def _linear_trend(values: np.ndarray, t_years: np.ndarray) -> Tuple[float, float]:
    """Ordinary least squares slope and intercept (ignoring NaNs)."""
    good = ~np.isnan(values)
    if good.sum() < 3:
        return 0.0, float(np.nanmean(values)) if good.any() else 0.0
    slope, intercept = np.polyfit(t_years[good], values[good], 1)
    return float(slope), float(intercept)


def _find_event_runs(
    flag: np.ndarray,
    min_duration: int,
    gap_fill: int,
) -> List[Tuple[int, int]]:
    """Return list of (start_idx, end_idx_inclusive) for runs >= min_duration.

    Adjacent runs separated by <= gap_fill timesteps are merged (Hobday style).
    """
    runs: List[List[int]] = []
    i = 0
    n = len(flag)
    while i < n:
        if flag[i]:
            s = i
            while i < n and flag[i]:
                i += 1
            runs.append([s, i - 1])
        else:
            i += 1

    if gap_fill > 0 and len(runs) > 1:
        merged = [runs[0]]
        for r in runs[1:]:
            if r[0] - merged[-1][1] - 1 <= gap_fill:
                merged[-1][1] = r[1]
            else:
                merged.append(list(r))
        runs = merged

    return [(s, e) for s, e in runs if (e - s + 1) >= min_duration]


def _event_statistics(
    values: np.ndarray,
    climatology: np.ndarray,
    threshold: np.ndarray,
    time: pd.DatetimeIndex,
    runs: List[Tuple[int, int]],
    direction: str,
) -> pd.DataFrame:
    """Compute per-event statistics and Hobday category."""
    rows = []
    sign = 1 if direction == 'above' else -1
    for s, e in runs:
        vals = values[s:e + 1]
        clim = climatology[s:e + 1]
        thr = threshold[s:e + 1]
        # Anomaly and excess (signed so that "worse" is always positive)
        anom = sign * (vals - clim)
        excess = sign * (vals - thr)
        # Peak
        peak_idx = int(np.argmax(excess))
        i_max = float(anom[peak_idx])
        i_thresh_peak = float(sign * (thr[peak_idx] - clim[peak_idx]))
        cat_ratio = i_max / i_thresh_peak if i_thresh_peak > 0 else 1.0

        if cat_ratio < 2:
            category = 1
        elif cat_ratio < 3:
            category = 2
        elif cat_ratio < 4:
            category = 3
        else:
            category = 4

        rows.append({
            'start': time[s],
            'end': time[e],
            'start_year': int(time[s].year),
            'start_month': int(time[s].month),
            'duration': int(e - s + 1),
            'mean_anomaly': float(np.nanmean(anom)),
            'max_anomaly': i_max,
            'max_excess_above_threshold': float(np.nanmax(excess)),
            'cumulative_anomaly': float(np.nansum(anom)),
            'category': category,
            'category_ratio': round(cat_ratio, 3),
        })
    return pd.DataFrame(rows)


# ============================================================================
# HIGH-LEVEL DETECTOR
# ============================================================================

class DualBaselineDetector:
    """Detect threshold-exceedance events under both fixed and shifting baselines.

    Parameters
    ----------
    baseline_years : tuple of int
        (start_year, end_year) for the fixed baseline climatology, inclusive.
    percentile : float
        Percentile threshold (e.g. 90 for MHW, 10 for hypoxia/OA).
    min_duration : int
        Minimum event duration in time steps (5 days for daily SST, 3 months
        for monthly bottom temperature per Hobday 2016 / this paper's methods).
    gap_fill : int
        Merge adjacent events separated by <= gap_fill time steps.
    direction : {'above', 'below'}
        'above' for heatwaves (T > p90), 'below' for hypoxia / OA (O2 < p10,
        pH < p10). For 'below' use `percentile=10` (the tool will compute the
        lower-tail threshold correctly).
    half_window_days : int
        DOY pooling half-window (5 => ±5 days, as in Hobday 2016). Only used
        when time_unit='day'.
    smooth_window : int
        Climatology smoothing window in time steps (31 days for daily data).
    time_unit : {'day', 'month'}
        Temporal resolution of the input data.

    Notes
    -----
    The shifting-baseline threshold is constructed as:
        thresh_shift(t) = thresh_fixed(doy(t)) + slope * (t - t_baseline_mean)
    where slope is the linear trend in the variable over the full record.
    This is equivalent to detrending the data before applying the fixed
    threshold, and matches the operational definition recommended by
    Amaya et al. (2023).
    """

    def __init__(
        self,
        baseline_years: Tuple[int, int] = (2017, 2022),
        percentile: float = 90,
        min_duration: int = 5,
        gap_fill: int = 2,
        direction: str = 'above',
        half_window_days: int = 5,
        smooth_window: int = 31,
        time_unit: str = 'day',
    ):
        if direction not in ('above', 'below'):
            raise ValueError("direction must be 'above' or 'below'")
        if time_unit not in ('day', 'month'):
            raise ValueError("time_unit must be 'day' or 'month'")
        if direction == 'below' and percentile > 50:
            warnings.warn(
                f"direction='below' with percentile={percentile} looks unusual — "
                "for hypoxia/OA you probably want percentile=10.",
            )

        self.baseline_years = baseline_years
        self.percentile = percentile
        self.min_duration = min_duration
        self.gap_fill = gap_fill
        self.direction = direction
        self.half_window_days = half_window_days
        self.smooth_window = smooth_window
        self.time_unit = time_unit

    # ----- main entry point for 1D series -----

    def detect(
        self,
        data: Union[pd.Series, xr.DataArray, np.ndarray],
        time: Optional[pd.DatetimeIndex] = None,
        name: str = 'variable',
    ) -> DualResult:
        """Run dual-baseline detection on a 1D time series.

        Parameters
        ----------
        data : pd.Series (DatetimeIndex), xr.DataArray (with 'time' dim), or np.ndarray
            If np.ndarray, `time` must be provided.
        time : pd.DatetimeIndex, optional
            Required when `data` is a bare ndarray.
        name : str
            Label for the variable (used in returned metadata and plots).

        Returns
        -------
        DualResult
        """
        values, time_idx = self._coerce_input(data, time)

        t_years = (time_idx - time_idx[0]).days.values / 365.25
        baseline_mask = (
            (time_idx.year >= self.baseline_years[0])
            & (time_idx.year <= self.baseline_years[1])
        )
        if not baseline_mask.any():
            raise ValueError(
                f"No timesteps fall within baseline_years={self.baseline_years}"
            )

        # 1) Climatology + fixed threshold
        if self.time_unit == 'day':
            doy = time_idx.dayofyear.values
            clim_arr, thresh_arr = _doy_climatology(
                values, doy, baseline_mask, self.percentile,
                half_window_days=self.half_window_days,
                smooth_window=self.smooth_window,
            )
            fixed_clim = clim_arr[doy - 1]
            fixed_thresh = thresh_arr[doy - 1]
        else:  # monthly
            month = time_idx.month.values
            clim_arr, thresh_arr = _monthly_climatology(
                values, month, baseline_mask, self.percentile,
            )
            fixed_clim = clim_arr[month - 1]
            fixed_thresh = thresh_arr[month - 1]

        # 2) Linear trend for shifting baseline
        slope, _ = _linear_trend(values, t_years)
        # Offset so that over the baseline period, the shifting threshold equals
        # the fixed threshold on average — preserves the interpretation.
        baseline_trend_mean = slope * t_years[baseline_mask].mean()
        trend_offset = slope * t_years - baseline_trend_mean

        shift_clim = fixed_clim + trend_offset
        shift_thresh = fixed_thresh + trend_offset

        # 3) Detect events under both baselines
        fixed_res = self._events_from_threshold(
            values, fixed_clim, fixed_thresh, time_idx, baseline='fixed',
        )
        shift_res = self._events_from_threshold(
            values, shift_clim, shift_thresh, time_idx, baseline='shifting',
        )

        return DualResult(
            fixed=fixed_res,
            shifting=shift_res,
            variable=name,
            trend_per_year=slope,
            baseline_years=self.baseline_years,
        )

    # ----- gridded entry point -----

    def detect_gridded(
        self,
        data: xr.DataArray,
        time_dim: str = 'time',
    ) -> xr.Dataset:
        """Apply 1D detection to each grid cell of an xarray DataArray.

        Parameters
        ----------
        data : xr.DataArray
            Must have a `time` dimension plus any number of spatial dims.
        time_dim : str
            Name of the time dimension.

        Returns
        -------
        xr.Dataset with variables:
            flag_fixed, flag_shift : bool (time x space)
            thresh_fixed, thresh_shift : float (time x space)
            clim_fixed, clim_shift : float (time x space)
        Event tables are NOT returned per-cell (too large); compute them
        per-point using `detect()` on specific cells of interest.
        """
        if time_dim not in data.dims:
            raise ValueError(f"time dim '{time_dim}' not in data")

        time_idx = pd.DatetimeIndex(data[time_dim].values)
        t_years = (time_idx - time_idx[0]).days.values / 365.25
        baseline_mask = (
            (time_idx.year >= self.baseline_years[0])
            & (time_idx.year <= self.baseline_years[1])
        )

        # Precompute day-of-year / month arrays
        if self.time_unit == 'day':
            doy = time_idx.dayofyear.values
            key_arr = doy
            half_w = self.half_window_days
            smooth_w = self.smooth_window
        else:
            key_arr = time_idx.month.values
            half_w = 0
            smooth_w = 1

        def per_cell(v: np.ndarray) -> np.ndarray:
            """Return a 6 x T array: flag_f, flag_s, thresh_f, thresh_s, clim_f, clim_s."""
            if np.all(np.isnan(v)):
                T = len(v)
                return np.full((6, T), np.nan)
            if self.time_unit == 'day':
                c_arr, t_arr = _doy_climatology(
                    v, key_arr, baseline_mask, self.percentile,
                    half_window_days=half_w, smooth_window=smooth_w,
                )
                clim_f = c_arr[key_arr - 1]
                thr_f = t_arr[key_arr - 1]
            else:
                c_arr, t_arr = _monthly_climatology(
                    v, key_arr, baseline_mask, self.percentile,
                )
                clim_f = c_arr[key_arr - 1]
                thr_f = t_arr[key_arr - 1]

            slope, _ = _linear_trend(v, t_years)
            baseline_trend_mean = slope * t_years[baseline_mask].mean()
            trend_offset = slope * t_years - baseline_trend_mean
            clim_s = clim_f + trend_offset
            thr_s = thr_f + trend_offset

            if self.direction == 'above':
                f_f = (v > thr_f).astype(float)
                f_s = (v > thr_s).astype(float)
            else:
                f_f = (v < thr_f).astype(float)
                f_s = (v < thr_s).astype(float)

            return np.stack([f_f, f_s, thr_f, thr_s, clim_f, clim_s])

        result = xr.apply_ufunc(
            per_cell,
            data,
            input_core_dims=[[time_dim]],
            output_core_dims=[['_stat', time_dim]],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float],
            dask_gufunc_kwargs={'output_sizes': {'_stat': 6}, 'allow_rechunk': True},
        )

        ds = xr.Dataset({
            'flag_fixed':   result.isel(_stat=0).astype(bool),
            'flag_shift':   result.isel(_stat=1).astype(bool),
            'thresh_fixed': result.isel(_stat=2),
            'thresh_shift': result.isel(_stat=3),
            'clim_fixed':   result.isel(_stat=4),
            'clim_shift':   result.isel(_stat=5),
        })
        ds.attrs.update({
            'percentile': self.percentile,
            'baseline_years': f"{self.baseline_years[0]}-{self.baseline_years[1]}",
            'direction': self.direction,
            'min_duration': self.min_duration,
            'gap_fill': self.gap_fill,
            'time_unit': self.time_unit,
        })
        return ds

    # ----- helpers -----

    def _coerce_input(
        self,
        data: Union[pd.Series, xr.DataArray, np.ndarray],
        time: Optional[pd.DatetimeIndex],
    ) -> Tuple[np.ndarray, pd.DatetimeIndex]:
        if isinstance(data, pd.Series):
            if not isinstance(data.index, pd.DatetimeIndex):
                raise ValueError("pd.Series input must have a DatetimeIndex")
            return data.values.astype(float), data.index
        if isinstance(data, xr.DataArray):
            if 'time' not in data.dims:
                raise ValueError("xr.DataArray input must have a 'time' dim")
            if data.ndim != 1:
                raise ValueError("Use detect_gridded() for multidim DataArrays")
            return data.values.astype(float), pd.DatetimeIndex(data.time.values)
        if isinstance(data, np.ndarray):
            if time is None:
                raise ValueError("time kwarg required for ndarray input")
            return data.astype(float), pd.DatetimeIndex(time)
        raise TypeError(f"Unsupported input type: {type(data)}")

    def _events_from_threshold(
        self,
        values: np.ndarray,
        clim: np.ndarray,
        thresh: np.ndarray,
        time_idx: pd.DatetimeIndex,
        baseline: str,
    ) -> DetectionResult:
        if self.direction == 'above':
            flag_arr = values > thresh
        else:
            flag_arr = values < thresh

        runs = _find_event_runs(flag_arr, self.min_duration, self.gap_fill)
        events_df = _event_statistics(
            values, clim, thresh, time_idx, runs, self.direction,
        )
        return DetectionResult(
            events=events_df,
            flag=pd.Series(flag_arr, index=time_idx, name='flag'),
            threshold=pd.Series(thresh, index=time_idx, name='threshold'),
            climatology=pd.Series(clim, index=time_idx, name='climatology'),
            baseline=baseline,
        )


# ============================================================================
# CHRONICITY METRICS
# ============================================================================

class ChronicityMetrics:
    """Diagnostics for when a stressor transitions from episodic to chronic."""

    @staticmethod
    def annual_exposure(flag: pd.Series) -> pd.Series:
        """Fraction of each calendar year spent in-event (0..100 %)."""
        if not isinstance(flag.index, pd.DatetimeIndex):
            raise ValueError("flag must be indexed by DatetimeIndex")
        return flag.groupby(flag.index.year).mean() * 100.0

    @staticmethod
    def chronic_transition_year_from_arrays(
        annual_pct: np.ndarray,
        years: np.ndarray,
        threshold: float = 50.0,
        run: int = 3,
    ) -> float:
        """Vector-friendly twin of ``chronic_transition_year`` (returns NaN if never)."""
        vals = np.asarray(annual_pct, dtype=float)
        yrs = np.asarray(years)
        if len(vals) < run:
            return float('nan')
        for i in range(len(vals) - run + 1):
            if np.all(vals[i:i + run] >= threshold):
                return float(yrs[i])
        return float('nan')

    @staticmethod
    def chronic_transition_year(
        annual_exposure: pd.Series,
        threshold: float = 50.0,
        run: int = 3,
    ) -> Optional[int]:
        """First year starting a `run` of consecutive years with exposure >= threshold.

        Returns None if no such run exists in the record.
        """
        y = ChronicityMetrics.chronic_transition_year_from_arrays(
            annual_exposure.values.astype(float),
            annual_exposure.index.values,
            threshold,
            run,
        )
        if not np.isfinite(y):
            return None
        return int(y)

    @staticmethod
    def chronicity_index(flag: pd.Series) -> float:
        """Ratio of longest continuous event duration to total record length.

        Matches the `chronicity index` referenced in QIN v6 Results. CI=1 means
        the longest event spans the entire record (fully chronic); CI near 0
        means even the longest event is brief relative to the record.
        """
        arr = flag.values.astype(bool)
        n = len(arr)
        if n == 0:
            return 0.0
        longest = 0
        cur = 0
        for v in arr:
            if v:
                cur += 1
                longest = max(longest, cur)
            else:
                cur = 0
        return longest / n

    @staticmethod
    def mean_return_interval(events: pd.DataFrame) -> float:
        """Mean gap (in the original time unit) between consecutive events.

        Returns NaN if fewer than 2 events.
        """
        if len(events) < 2:
            return float('nan')
        # Sort by start and compute gap between end_i and start_{i+1}
        ev = events.sort_values('start').reset_index(drop=True)
        gaps = (ev['start'].iloc[1:].values - ev['end'].iloc[:-1].values)
        gaps_days = pd.to_timedelta(gaps).days.astype(float)
        return float(np.nanmean(gaps_days))


# ============================================================================
# CASCADE FIDELITY
# ============================================================================

class CascadeFidelity:
    """Driver -> response cascade lift (odds ratio).

    For each rolling time window, compute:
        lift = P(response | driver occurred in last `window` timesteps)
               ---------------------------------------------------------
                            P(response at any time)

    Interpretation:
        lift ~ 1  : response co-occurrence with driver is no better than chance
        lift > 1  : genuine cascade signal
        lift >> 1 : strong cascade signal
        lift << 1 : response is *less* likely soon after a driver extreme than its
                    marginal rate in the same window (suppression / negative lift),
                    not merely “no cascade”.

    Crucially, under a fixed baseline, as both driver and response saturate
    toward "always in event", lift -> 1 even when the physical mechanism is
    intact. Under a shifting baseline the statistic remains informative.
    """

    @staticmethod
    def lift(
        driver_flag: pd.Series,
        response_flag: pd.Series,
        window_days: int = 30,
        block_years: int = 5,
    ) -> pd.DataFrame:
        """Compute cascade lift in rolling `block_years` windows.

        Parameters
        ----------
        driver_flag, response_flag : pd.Series[bool] with DatetimeIndex
        window_days : int
            Look-back window defining "driver occurred recently" **in time steps**
            along ``driver_flag`` (not calendar days unless the index is daily).
            For **monthly** coastal flags, use ``window_days=2`` to mean a 2-month
            look-back (same convention as ``_main_washington``).
        block_years : int
            Width of each analysis block **in calendar years** over which ``lift``
            is averaged (larger = smoother panel-(c) curves; try 1–2 for more
            year-to-year detail, at the cost of noise).

        Returns
        -------
        pd.DataFrame with columns:
            center_year, p_response, p_response_given_driver_recent, lift, n_timesteps
        """
        if not driver_flag.index.equals(response_flag.index):
            # Align
            common = driver_flag.index.intersection(response_flag.index)
            driver_flag = driver_flag.loc[common]
            response_flag = response_flag.loc[common]

        time_idx = driver_flag.index
        drv = driver_flag.values.astype(bool)
        rsp = response_flag.values.astype(bool)
        years = time_idx.year.values

        # Rolling "driver in last W timesteps" — assumes ~1-day resolution.
        # For monthly data, `window_days` should be set to the intended
        # look-back in months.
        driver_recent = np.zeros_like(drv)
        if window_days > 0 and len(drv) > window_days:
            # Cumulative-sum trick for rolling window
            csum = np.concatenate([[0], np.cumsum(drv.astype(int))])
            rolling = csum[window_days + 1:] - csum[:-window_days - 1]
            # `rolling[i]` = count of driver events in indices [i..i+window_days]
            # We want: driver occurred in strictly-prior window at index i
            # i.e. sum over (i-window_days) to (i-1)
            driver_recent = np.zeros_like(drv)
            for i in range(1, len(drv)):
                lo = max(0, i - window_days)
                driver_recent[i] = csum[i] - csum[lo] > 0

        yr_min, yr_max = int(years.min()), int(years.max())
        rows = []
        for yr_start in range(yr_min, yr_max - block_years + 2, block_years):
            mask = (years >= yr_start) & (years < yr_start + block_years)
            if mask.sum() == 0:
                continue
            p_resp = rsp[mask].mean()
            d_recent = driver_recent[mask]
            if d_recent.any():
                p_cond = rsp[mask][d_recent].mean()
            else:
                p_cond = np.nan
            lift = p_cond / p_resp if p_resp > 0 else np.nan
            rows.append({
                'center_year': yr_start + block_years / 2,
                'yr_start': yr_start,
                'yr_end': yr_start + block_years - 1,
                'p_response': float(p_resp),
                'p_response_given_driver_recent': float(p_cond),
                'lift': float(lift),
                'n_timesteps': int(mask.sum()),
            })
        return pd.DataFrame(rows)


# --- Panel (c): same gridded cascading definition as dual_baseline_sequential_cascade_stats.py
_DISPLAY_STRESSOR_TO_COMPOUND: Dict[str, str] = {
    'SHW': 'surface_heatwave',
    'BHW': 'bottom_heatwave',
    'Hypoxia': 'hypoxia',
    'OA': 'acidification',
}


def _compound_cascading_result_key(drv_display: str, rsp_display: str) -> str:
    a = _DISPLAY_STRESSOR_TO_COMPOUND[drv_display]
    b = _DISPLAY_STRESSOR_TO_COMPOUND[rsp_display]
    return f'cascading_{a}_triggers_{b}'


def _compound_cascade_block_percentages(
    cascade_da: xr.DataArray,
    ocean_mask: np.ndarray,
    block_years: int,
) -> pd.DataFrame:
    """Per-calendar block: mean(cascade mask) over ocean × time in block × 100.

    Matches the space–time average in ``dual_baseline_sequential_cascade_stats.domain_mean_frequency``,
    evaluated on each ``block_years`` window (for a time-resolved curve).
    """
    if 'time' not in cascade_da.dims:
        return pd.DataFrame(columns=['center_year', 'pct'])
    da = cascade_da
    if hasattr(da.data, 'compute'):
        da = da.compute()
    lat_n = 'lat' if 'lat' in da.dims else 'latitude'
    lon_n = 'lon' if 'lon' in da.dims else 'longitude'
    m = xr.DataArray(
        ocean_mask.astype(bool),
        dims=(lat_n, lon_n),
        coords={lat_n: da[lat_n], lon_n: da[lon_n]},
    )
    masked = da.astype(float).where(m)
    years = da['time'].dt.year.values
    yr_min, yr_max = int(np.min(years)), int(np.max(years))
    rows: List[Dict[str, float]] = []
    for yr_start in range(yr_min, yr_max - block_years + 2, block_years):
        t_sel = (da.time.dt.year >= yr_start) & (da.time.dt.year < yr_start + block_years)
        if not bool(t_sel.any()):
            continue
        sub = masked.sel(time=t_sel)
        val = float(sub.mean(skipna=True)) * 100.0
        rows.append({'center_year': float(yr_start) + block_years / 2.0, 'pct': val})
    return pd.DataFrame(rows)


def _build_gridded_compound_cascade_bundle(
    scenario: str,
    hypoxia_mode: str,
) -> Tuple[Dict[str, xr.DataArray], Dict[str, xr.DataArray], np.ndarray]:
    """Gridded extremes + compound events (fixed/shifting) for panel (c)."""
    from advanced_stressor_metrics import AdvancedStressorMetrics

    from dual_baseline_sequential_cascade_stats import build_extremes_monthly
    from dual_baseline_spatial_emergence import load_gridded_monthly_bundle

    bundle = load_gridded_monthly_bundle(scenario=scenario)
    monthly = {
        k: (v.compute() if hasattr(v.data, 'compute') else v)
        for k, v in bundle['monthly'].items()  # type: ignore[index]
    }
    ocean_mask = bundle['ocean_mask']  # type: ignore[index]
    metrics = AdvancedStressorMetrics(baseline_period=(2017, 2022))
    compound_fixed: Dict[str, xr.DataArray] = {}
    compound_shift: Dict[str, xr.DataArray] = {}
    for mode, out in (('fixed', compound_fixed), ('shifting', compound_shift)):
        extremes = build_extremes_monthly(
            monthly, ocean_mask, hypoxia_mode, mode,  # type: ignore[arg-type]
        )
        comp = metrics.classify_compound_events(
            extremes,
            temporal_window=1,
            lag_threshold=1,
            time_unit='month',
        )
        for k, v in comp.items():
            if not (k.startswith('cascading_') and isinstance(v, xr.DataArray)):
                continue
            out[k] = v.compute() if hasattr(v.data, 'compute') else v
    return compound_fixed, compound_shift, ocean_mask


# ============================================================================
# PLOTTING
# ============================================================================

def _tab10_color(i: int) -> Tuple[float, float, float, float]:
    """RGBA from matplotlib ``tab10`` for categorical styling."""
    import matplotlib.pyplot as plt

    cmap = plt.colormaps['tab10']
    return cmap(i % 10)


def _tab10_colors(n: int) -> List[Tuple[float, float, float, float]]:
    return [_tab10_color(i) for i in range(n)]


def _seaborn_tab10_new_palette(n: int) -> List[Union[str, Tuple[float, ...]]]:
    """Discrete colors from ``cmap.Colormap('seaborn:tab10_new')`` when installed."""
    try:
        from cmap import Colormap

        cm = Colormap('seaborn:tab10_new')
        return [cm(i % 10) for i in range(n)]
    except ImportError:
        import matplotlib.pyplot as plt

        tc = plt.colormaps['tab10']
        return [tc(i % 10) for i in range(n)]


def _seaborn_tab10_new_color(i: int) -> Union[str, Tuple[float, ...]]:
    try:
        from cmap import Colormap

        return Colormap('seaborn:tab10_new')(i % 10)
    except ImportError:
        import matplotlib.pyplot as plt

        return tuple(plt.colormaps['tab10'](i % 10))


def chronicity_panel_a_palette(
    n_stressors: int,
    n_cascade_pairs: int = 0,
) -> List[Union[str, Tuple[float, ...]]]:
    """Discrete colors for ``plot_chronicity_and_cascade`` panel (a) stressor lines.

    Must match the ``max(10, n_stressors + n_cascade_pairs)`` rule so hue *i* is the
    same in the chronicity figure and in ``plot_improved_figure1`` bar panel c.
    """
    return _seaborn_tab10_new_palette(max(10, int(n_stressors) + int(n_cascade_pairs)))


def chronicity_palette_stressor_rgb(
    stressor_index: int,
    palette: Sequence[Union[str, Tuple[float, ...]]],
) -> Tuple[float, float, float]:
    """RGB for stressor ``stressor_index`` — same saturated hue as panel (a) means."""
    import matplotlib.colors as mcolors

    return tuple(mcolors.to_rgb(palette[stressor_index % len(palette)]))


_CAT_NAMES = {1: 'Cat 1 Moderate', 2: 'Cat 2 Strong',
              3: 'Cat 3 Severe',    4: 'Cat 4 Extreme'}


def _cat_colors_hobday() -> Dict[int, Tuple[float, float, float]]:
    """Severity ramp aligned with Hobday et al. (2018) schematic (I–IV)."""
    import matplotlib.colors as mcolors

    hexes = {
        1: '#f7e98f',  # moderate — pale yellow–gold
        2: '#f0a64a',  # strong — orange
        3: '#d64545',  # severe — red
        4: '#5c0a1f',  # extreme — burgundy
    }
    return {k: mcolors.to_rgb(h) for k, h in hexes.items()}


def _apply_plot_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#333333', 'axes.linewidth': 0.6,
        'xtick.color': '#333333', 'ytick.color': '#333333',
        'font.family': 'DejaVu Sans',
    })


# Shared by ``plot_improved_figure1`` and ``plot_chronicity_and_cascade`` (chronicity panel a).
FIG_AXIS_LABEL_FONTSIZE = 13.5

# Readable on white for chronicity panel (a) regime label and panel (c) cascade hint.
CHRONIC_DIAGNOSTIC_TEXT_GREEN = '#14532d'
# Panel (a): chronic threshold dotted line (darker green).
CHRONIC_REGIME_LINE_GREEN = '#052a0c'


def plot_improved_figure1(
    result: DualResult,
    region_name: str = 'Washington coast',
    scenario: str = 'SSP2-4.5',
    artifact_threshold_days: int = 180,
    percentile: float = 90.0,
    figsize: Tuple[float, float] = (14, 9),
    panel_c_extras: Optional[Sequence[Tuple[str, DualResult]]] = None,
    panel_c_extras_only: bool = False,
    panel_c_chronicity_palette_ns: int = 4,
    panel_c_chronicity_palette_nc: int = 2,
) -> 'matplotlib.figure.Figure':
    """Recreate the 3-panel improved Figure 1 (duration–intensity scatter + exposure timeline).

    Panel ``c`` draws that reference only from the first year of the series through
    ``min(2060, last year)`` (no extension past 2060). It marks ``100 - percentile``
    percent of the year: for a stationary series and a well-calibrated upper-tail threshold
    (e.g. p90), the long-run fraction of timesteps above threshold should be about
    that value. Persistent elevation far above it under a fixed baseline flags the
    perpetual-exceedance / baseline-shift artifact.

    ``panel_c_extras`` can supply one ``DualResult`` (e.g. full-domain mean SST) so
    panel **c** is a grouped bar chart (fixed vs shifting per year) instead of the
    default exposure time series. Only the first extra entry is used.

    ``panel_c_chronicity_palette_ns`` / ``panel_c_chronicity_palette_nc`` should match
    ``plot_chronicity_and_cascade`` (stressors + cascade pairs) so panel (c) bar
    hues align with the same tab10_new indices as panel (a) for the fixed series.
    """
    import matplotlib.pyplot as plt
    _apply_plot_style()

    df_f = result.fixed.events
    df_s = result.shifting.events
    flag_f = result.fixed.flag
    flag_s = result.shifting.flag

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.38, wspace=0.22)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax3 = fig.add_subplot(gs[1, :])

    cat_colors = _cat_colors_hobday()

    def scatter(
        ax,
        df,
        show_artifact=False,
        label_extremes_by: str = 'max_anomaly',
    ):
        if df.empty:
            ax.text(0.5, 0.5, 'No events detected',
                    ha='center', va='center', transform=ax.transAxes)
            return
        df = df.sort_values('category')
        for cat in sorted(df['category'].unique()):
            sub = df[df['category'] == cat]
            ax.scatter(
                sub['duration'], sub['max_anomaly'],
                s=68 + sub['category_ratio'] * 14,
                color=cat_colors[cat],
                edgecolors='#222222', linewidths=0.4,
                alpha=0.78, label=_CAT_NAMES[cat], zorder=3,
            )
        n_label = min(8, len(df))
        if label_extremes_by == 'duration':
            top = df.nlargest(n_label, 'duration').sort_values('max_anomaly')
            offsets = [
                (-32, 4), (-8, -14), (10, 6), (8, -12),
                (-40, -6), (14, 10), (-22, 14), (18, -16),
            ]
        else:
            top = df.nlargest(n_label, 'max_anomaly').sort_values('duration')
            offsets = [
                (6, 5), (6, -12), (6, 5), (-30, 5),
                (-8, -14), (12, 8), (-34, -4), (10, -10),
            ]
        for (_, e), (dx, dy) in zip(top.iterrows(), offsets):
            ax.annotate(str(e['start_year']),
                        (e['duration'], e['max_anomaly']),
                        xytext=(dx, dy), textcoords='offset points',
                        fontsize=8, color='#333333')
        ax.set_xlabel('Event duration (time steps)', fontsize=FIG_AXIS_LABEL_FONTSIZE)
        ax.set_ylabel('Max anomaly', fontsize=FIG_AXIS_LABEL_FONTSIZE)
        if show_artifact and len(df):
            xmax = df['duration'].max()
            ax.axvspan(
                artifact_threshold_days, xmax * 1.15,
                facecolor='lightgrey', edgecolor='none', alpha=0.55, zorder=1,
            )
        ax.grid(True, linestyle=':', linewidth=0.4, color='#bbbbbb', alpha=0.6)
        ax.set_axisbelow(True)

    scatter(ax1, df_f, show_artifact=True)
    scatter(ax2, df_s, label_extremes_by='duration')

    if not df_f.empty:
        ax1.legend(
            loc='lower right', frameon=True, framealpha=0.95,
            edgecolor='#bbbbbb', fontsize=8.5,
            title='Hobday category', title_fontsize=8.5,
        )

    # Panel c: domain aggregate time series and/or optional bar-chart ``DualResult``
    c_fixed = cat_colors[3]   # Cat 3 Severe
    c_shift = 'tab:orange'    # shifting baseline (hue distinct from fixed bars / fill)
    exp_f = ChronicityMetrics.annual_exposure(flag_f)
    exp_s = ChronicityMetrics.annual_exposure(flag_s)
    panel_c_ymax = 100.0
    yr_lo = int(min(exp_f.index.min(), exp_s.index.min()))
    yr_ref_end = int(min(2060, max(exp_f.index.max(), exp_s.index.max())))

    def _bar_year_ticks(ax_bar, x_pos: np.ndarray, years_list: List[int], n_yr: int) -> None:
        ax_bar.set_xticks(x_pos)
        if n_yr > 20:
            step = max(1, n_yr // 14)
            ax_bar.set_xticks(x_pos[::step])
            ax_bar.set_xticklabels([str(years_list[i]) for i in range(0, n_yr, step)])
        else:
            ax_bar.set_xticklabels([str(y) for y in years_list])
        plt.setp(ax_bar.get_xticklabels(), rotation=40, ha='right')
        ax_bar.set_xlim(-0.5, n_yr - 0.5)

    if panel_c_extras_only and panel_c_extras:
        _pal_c = chronicity_panel_a_palette(
            panel_c_chronicity_palette_ns, panel_c_chronicity_palette_nc,
        )
        (_, r0) = panel_c_extras[0]
        s_f = ChronicityMetrics.annual_exposure(r0.fixed.flag)
        s_s = ChronicityMetrics.annual_exposure(r0.shifting.flag)
        idx_union = s_f.index.union(s_s.index)
        years = [int(y) for y in sorted(idx_union)]
        n = len(years)
        idx = pd.Index(years, name='year')
        h_f = s_f.reindex(idx).fillna(0.0).values.astype(float)
        h_s = s_s.reindex(idx).fillna(0.0).values.astype(float)
        x = np.arange(n, dtype=float)
        _slot_fill = 0.99
        _gap_b = 0.012
        w = (_slot_fill - _gap_b) / 2.0
        _m = (1.0 - _slot_fill) / 2.0
        x_f = x - 0.5 + _m
        x_s = x_f + w + _gap_b
        c0 = chronicity_palette_stressor_rgb(0, _pal_c)
        ax3.bar(
            x_f, h_f, width=w, facecolor=c0, edgecolor='#1a1a1a', lw=0.45,
            hatch='', label='Fixed', zorder=4,
        )
        ax3.bar(
            x_s, h_s, width=w, facecolor='tab:orange', edgecolor='#1a1a1a', lw=0.45,
            hatch='', label='Shifting', zorder=3,
        )
        ax3.set_title(
            str(panel_c_extras[0][0]), fontsize=11, loc='left', color='#1a1a1a',
        )
        yr_lo = years[0]
        yr_ref_end = min(2060, years[-1])
        panel_c_ymax = float(min(120.0, max(100.0, max(h_f.max(), h_s.max()) * 1.08)))
        _bar_year_ticks(ax3, x, years, n)
    else:
        ax3.fill_between(exp_f.index, 0, exp_f.values, color=c_fixed,
                         alpha=0.32, zorder=2, label='Fixed baseline')
        ax3.plot(exp_f.index, exp_f.values, '-', color=c_fixed, lw=1.5, zorder=3)
        ax3.fill_between(
            exp_s.index, 0, exp_s.values, facecolor=c_shift, edgecolor='#1a1a1a',
            hatch='', linewidth=0.35, alpha=0.42, zorder=4, label='Shifting baseline',
        )
        ax3.plot(exp_s.index, exp_s.values, '-', color=c_shift, lw=1.5, zorder=5)
        yr_lo = int(min(exp_f.index.min(), exp_s.index.min()))
        yr_ref_end = int(min(2060, max(exp_f.index.max(), exp_s.index.max())))
        panel_c_ymax = 100.0

    ref = 100.0 - float(percentile)
    ref_lbl = f'Stationary reference ({ref:.0f}% ≈ 100 − p{percentile:.0f})'

    if panel_c_extras_only and panel_c_extras:
        ax3.axhline(
            ref, ls='--', color='#1a1a1a', lw=1.25, zorder=6, label=ref_lbl,
        )
    else:
        ax3.plot(
            [yr_lo, yr_ref_end], [ref, ref],
            ls='--', color='#1a1a1a', lw=1.25, zorder=6, label=ref_lbl,
        )
    ax3.text(
        0.98, 0.94,
        f'Under a stationary climate, annual exposure should\n'
        f'sit near ~{ref:.0f}% for a calibrated p{percentile:.0f} threshold.',
        transform=ax3.transAxes, ha='right', va='top',
        fontsize=9.5, color='#0d0d0d',
        bbox=dict(
            boxstyle='round,pad=0.45', fc='white', ec='#333333',
            lw=0.9, alpha=0.96,
        ),
        zorder=7,
    )
    ax3.set_xlabel('Year', fontsize=FIG_AXIS_LABEL_FONTSIZE)
    if panel_c_extras_only and panel_c_extras:
        ax3.set_ylabel(
            '% of year above threshold\n(domain grid mean; annual % above threshold)',
            fontsize=FIG_AXIS_LABEL_FONTSIZE,
        )
    else:
        ax3.set_ylabel('% of year above threshold', fontsize=FIG_AXIS_LABEL_FONTSIZE)
    _leg_ncol = 2 if (panel_c_extras_only and panel_c_extras) else 1
    _leg_fs = 7.8 if (panel_c_extras_only and panel_c_extras) else 9
    ax3.legend(
        loc='upper left', frameon=True, framealpha=0.95,
        edgecolor='#bbbbbb', fontsize=_leg_fs, ncol=_leg_ncol,
        columnspacing=0.9, handlelength=2.2,
    )
    ax3.set_ylim(0, panel_c_ymax)
    ax3.grid(True, linestyle=':', linewidth=0.4, color='#bbbbbb', alpha=0.6)
    ax3.set_axisbelow(True)

    return fig


def plot_chronicity_and_cascade(
    stressor_results: Dict[str, DualResult],
    cascade_pairs: Optional[Dict[str, Tuple[str, str]]] = None,
    region_name: str = 'Washington coast',
    chronic_exposure_threshold: float = 50.0,
    chronic_run_years: int = 3,
    cascade_window_days: int = 30,
    cascade_block_years: int = 5,
    exposure_roll_years: int = 5,
    figsize: Tuple[float, float] = (14, 12.4),
    panel_c_gridded_cascade: Optional[
        Tuple[Dict[str, xr.DataArray], Dict[str, xr.DataArray], np.ndarray]
    ] = None,
) -> 'matplotlib.figure.Figure':
    """Chronicity transition + cascade fidelity figure (proposed new analysis).

    Parameters
    ----------
    stressor_results : dict[str, DualResult]
        Keys are stressor labels (e.g. 'SHW', 'BHW', 'Hypoxia', 'OA'); values
        are DualResult objects from DualBaselineDetector.detect().
    cascade_pairs : dict[str, tuple[str, str]], optional
        Map of cascade_name -> (driver_key, response_key). Example:
            {'BHW → Hypoxia': ('BHW', 'Hypoxia'),
             'BHW → OA': ('BHW', 'OA'),
             'SHW → OA': ('SHW', 'OA')}
        Each key must exist in ``stressor_results``.
        When the same ``drv_key`` appears in multiple pairs (e.g. BHW→Hypoxia and
        BHW→OA), the *n*th pair uses palette index ``driver_index + (n-1)`` so
        traces are visually distinct.
    panel_c_gridded_cascade : tuple or None, optional
        If set, panel **(c)** plots **gridded** ``cascading_*`` masks from
        ``AdvancedStressorMetrics.classify_compound_events`` (same pipeline as
        ``dual_baseline_sequential_cascade_stats.py``): per-block mean % of
        ocean×time in a cascading state. Tuple is
        ``(compound_fixed, compound_shifting, ocean_mask)`` with only keys
        starting with ``cascading_`` required. If ``None``, panel **(c)** uses
        ``CascadeFidelity.lift`` on the 1D coastal flags (legacy).
    exposure_roll_years : int
        Panel **(a)** only: centered rolling window (years) for annual exposure
        mean ± 1σ (default: 5).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    _apply_plot_style()

    cascade_pairs = cascade_pairs or {}

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1.95 * 1.2, 2.05],
        hspace=0.26,
        wspace=0.28,
        left=0.13,
        right=0.99,
        top=0.92,
        bottom=0.07,
    )
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    palette = chronicity_panel_a_palette(
        len(stressor_results), len(cascade_pairs or {}),
    )

    def _rolling_mean_std_band(series: pd.Series, win: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Centered rolling mean and (mean ± std) of annual exposure on ``series`` index."""
        s = series.sort_index()
        m = s.rolling(window=win, center=True, min_periods=1).mean()
        sd = s.rolling(window=win, center=True, min_periods=1).std()
        sd = sd.fillna(0.0)
        lo = (m - sd).clip(lower=0.0, upper=100.0)
        hi = (m + sd).clip(lower=0.0, upper=100.0)
        return m, lo, hi

    yr_min = int(min(r.fixed.flag.index.year.min() for r in stressor_results.values()))
    yr_max = int(max(r.fixed.flag.index.year.max() for r in stressor_results.values()))

    # Panel A: rolling mean ± 1σ (annual %) + mean lines (solid=fixed, dashed=shifting)
    _a_ts_lw_fixed = 4.25
    _a_ts_lw_shift = 3.75
    chronic_yrs: Dict[str, Dict[str, Optional[int]]] = {}
    for i, (key, result) in enumerate(stressor_results.items()):
        c = palette[i % len(palette)]
        exp_f = ChronicityMetrics.annual_exposure(result.fixed.flag)
        exp_s = ChronicityMetrics.annual_exposure(result.shifting.flag)
        mean_f, lo_f, hi_f = _rolling_mean_std_band(exp_f, exposure_roll_years)
        mean_s, lo_s, hi_s = _rolling_mean_std_band(exp_s, exposure_roll_years)

        ax_a.fill_between(
            mean_f.index, lo_f.values, hi_f.values,
            color=c, alpha=0.22, zorder=2, linewidth=0, label='_nolegend_',
        )
        ax_a.fill_between(
            mean_s.index, lo_s.values, hi_s.values,
            color=c, alpha=0.14, zorder=2, linewidth=0, label='_nolegend_',
        )
        ax_a.plot(mean_f.index, mean_f.values, '-', color=c, lw=_a_ts_lw_fixed, zorder=5, label=key)
        ax_a.plot(
            mean_s.index, mean_s.values, '--', color=c, lw=_a_ts_lw_shift,
            alpha=0.9, zorder=5, dashes=(6, 2.5), label='_nolegend_',
        )
        chronic_yrs[key] = {
            'fixed': ChronicityMetrics.chronic_transition_year(
                exp_f, chronic_exposure_threshold, chronic_run_years,
            ),
            'shifting': ChronicityMetrics.chronic_transition_year(
                exp_s, chronic_exposure_threshold, chronic_run_years,
            ),
        }

    _chronic_c = _seaborn_tab10_new_color(3)
    ax_a.axhline(
        chronic_exposure_threshold, ls=':', color=CHRONIC_REGIME_LINE_GREEN,
        lw=2.75, zorder=3, dash_capstyle='round',
    )
    _a_chronic_label_fs = 15

    ax_a.set_xlabel('Year', fontsize=FIG_AXIS_LABEL_FONTSIZE)
    ax_a.set_ylabel('Annual exposure (% of year)', fontsize=FIG_AXIS_LABEL_FONTSIZE)
    ax_a.set_ylim(0, 100)
    ax_a.set_xlim(yr_min, yr_max)
    ax_a.tick_params(axis='both', labelsize=12)

    legend_entries = [
        Line2D([0], [0], color='#333333', lw=_a_ts_lw_fixed, ls='-',
               label=f'Fixed ({exposure_roll_years}-yr mean)'),
        Line2D([0], [0], color='#333333', lw=_a_ts_lw_shift, ls='--',
               label=f'Shifting ({exposure_roll_years}-yr mean)'),
        Patch(facecolor='#888888', edgecolor='none', alpha=0.35,
              label=f'Mean ± 1σ ({exposure_roll_years}-yr window)'),
    ]
    for i, key in enumerate(stressor_results):
        legend_entries.append(
            Line2D([0], [0], color=palette[i % len(palette)], lw=_a_ts_lw_fixed, ls='-', label=key),
        )
    ax_a.legend(
        handles=legend_entries, loc='upper left', frameon=True,
        framealpha=0.95, edgecolor='#bbbbbb', fontsize=11.5,
        ncol=2, columnspacing=1.0,
    )
    ax_a.text(
        list(stressor_results.values())[0].fixed.flag.index.year.min() + 1,
        chronic_exposure_threshold + 3,
        f'chronic regime (>{chronic_exposure_threshold:.0f}% of year)',
        fontsize=_a_chronic_label_fs, color=CHRONIC_DIAGNOSTIC_TEXT_GREEN, style='italic',
        zorder=25,
    )
    ax_a.grid(True, linestyle=':', linewidth=0.4, color='#bbbbbb', alpha=0.5)
    ax_a.set_axisbelow(True)

    # Panel B: chronic transition year bar chart
    _bc_fs = 11.5
    keys = list(stressor_results.keys())
    x = np.arange(len(keys))
    w = 0.36
    for i, key in enumerate(keys):
        c = palette[i % len(palette)]
        fy = chronic_yrs[key]['fixed']
        sy = chronic_yrs[key]['shifting']

        if fy is not None:
            ax_b.bar(i - w / 2, fy - yr_min, w, bottom=yr_min,
                     color=c, edgecolor='#222222', lw=0.5)
            ax_b.text(i - w / 2, fy + 0.5, str(fy), ha='center', fontsize=_bc_fs, color=c)
        else:
            ax_b.text(i - w / 2, yr_max + 0.5, f'>{yr_max}', ha='center',
                      fontsize=_bc_fs - 0.5, color=c, style='italic')
        if sy is not None:
            ax_b.bar(i + w / 2, sy - yr_min, w, bottom=yr_min,
                     color=c, hatch='//', edgecolor='#222222', lw=0.5, alpha=0.5)
            ax_b.text(i + w / 2, sy + 0.5, str(sy), ha='center', fontsize=_bc_fs, color=c)
        else:
            ax_b.text(i + w / 2, yr_max + 0.5, 'never\n(in record)', ha='center',
                      fontsize=_bc_fs - 0.5, color='#666666', style='italic')

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(keys, fontsize=_bc_fs)
    ax_b.set_ylabel('Year of transition to chronic', fontsize=FIG_AXIS_LABEL_FONTSIZE)
    ax_b.set_ylim(yr_min, yr_max + 8)
    ax_b.tick_params(axis='y', labelsize=_bc_fs)
    ax_b.grid(True, axis='y', linestyle=':', linewidth=0.45, color='#bbbbbb', alpha=0.5)
    _leg = _seaborn_tab10_new_color(7)
    ax_b.legend(handles=[
        Patch(facecolor=_leg, edgecolor='#222222', lw=0.5, label='Fixed'),
        Patch(facecolor=_leg, edgecolor='#222222', lw=0.5,
              hatch='//', alpha=0.5, label='Shifting'),
    ], loc='upper right', frameon=True, framealpha=0.95, fontsize=_bc_fs)

    # Panel C: cascade lift (1D) OR gridded cascading % (same as sequential_cascade_stats)
    if cascade_pairs:
        def _cascade_legend_label(full_name: str) -> str:
            """Compact labels for panel (c) legend (e.g. acidification -> OA)."""
            return full_name.replace('acidification', 'OA')

        markers = ['o', '^', 's', 'D', 'v']
        _cascade_color_off: Dict[str, int] = {}
        stressor_keys = list(stressor_results.keys())
        cascade_legend_handles: List[Line2D] = []

        if panel_c_gridded_cascade is not None:
            compound_f, compound_s, ocean_mask = panel_c_gridded_cascade
            _pct_sample: List[float] = []
            for i, (name, (drv_key, rsp_key)) in enumerate(cascade_pairs.items()):
                if drv_key not in stressor_results or rsp_key not in stressor_results:
                    continue
                ck = _compound_cascading_result_key(drv_key, rsp_key)
                if ck not in compound_f or ck not in compound_s:
                    continue
                di = stressor_keys.index(drv_key)
                off = _cascade_color_off.get(drv_key, 0)
                c = palette[(di + off) % len(palette)]
                _cascade_color_off[drv_key] = off + 1
                m = markers[i % len(markers)]
                df_f = _compound_cascade_block_percentages(
                    compound_f[ck], ocean_mask, cascade_block_years,
                )
                df_s = _compound_cascade_block_percentages(
                    compound_s[ck], ocean_mask, cascade_block_years,
                )
                _pct_sample.extend(df_f['pct'].astype(float).tolist())
                _pct_sample.extend(df_s['pct'].astype(float).tolist())
                ax_c.plot(
                    df_f['center_year'], df_f['pct'],
                    f'{m}-', color=c, lw=3.0, ms=8,
                    markeredgecolor='#222222', markeredgewidth=0.55,
                    label='_nolegend_',
                )
                ax_c.plot(
                    df_s['center_year'], df_s['pct'],
                    f'{m}--', color=c, lw=2.55, ms=7, alpha=0.78,
                    markeredgecolor='#222222', markeredgewidth=0.55,
                    dashes=(5, 2.5), label='_nolegend_',
                )
                cascade_legend_handles.append(
                    Line2D(
                        [0], [0], color=c, ls='-', lw=3.0, marker=m, ms=8,
                        markeredgecolor='#222222', markeredgewidth=0.55,
                        label=_cascade_legend_label(name),
                    ),
                )
            fin = [float(x) for x in _pct_sample if np.isfinite(x)]
            hi = max(fin) if fin else 0.0
            ymax = max(hi * 1.12 + 1e-9, 0.05)
            ax_c.set_ylim(0.0, ymax)
            ax_c.set_xlabel(
                f'Year (centre of {cascade_block_years}-yr block)',
                fontsize=FIG_AXIS_LABEL_FONTSIZE,
            )
            ax_c.set_ylabel(
                'Cascading compound events\n'
                '(% ocean–time in block)\n'
                '(same pipeline as dual_baseline_sequential_cascade_stats)',
                fontsize=FIG_AXIS_LABEL_FONTSIZE,
                labelpad=8,
            )
            _c_panel_note = (
                'Gridded cascading\n(1-mo lag; intensity criterion)',
                CHRONIC_DIAGNOSTIC_TEXT_GREEN,
            )
        else:
            _lift_sample: List[float] = []
            for i, (name, (drv_key, rsp_key)) in enumerate(cascade_pairs.items()):
                if drv_key not in stressor_results or rsp_key not in stressor_results:
                    continue
                di = stressor_keys.index(drv_key)
                off = _cascade_color_off.get(drv_key, 0)
                c = palette[(di + off) % len(palette)]
                _cascade_color_off[drv_key] = off + 1
                m = markers[i % len(markers)]

                lift_f = CascadeFidelity.lift(
                    stressor_results[drv_key].fixed.flag,
                    stressor_results[rsp_key].fixed.flag,
                    window_days=cascade_window_days,
                    block_years=cascade_block_years,
                )
                lift_s = CascadeFidelity.lift(
                    stressor_results[drv_key].shifting.flag,
                    stressor_results[rsp_key].shifting.flag,
                    window_days=cascade_window_days,
                    block_years=cascade_block_years,
                )
                for col in (lift_f['lift'], lift_s['lift']):
                    v = col.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
                    _lift_sample.extend(v.tolist())

                ax_c.plot(lift_f['center_year'], lift_f['lift'],
                           f'{m}-', color=c, lw=3.0, ms=8,
                           markeredgecolor='#222222', markeredgewidth=0.55,
                           label='_nolegend_')
                ax_c.plot(lift_s['center_year'], lift_s['lift'],
                           f'{m}--', color=c, lw=2.55, ms=7, alpha=0.78,
                           markeredgecolor='#222222', markeredgewidth=0.55,
                           dashes=(5, 2.5), label='_nolegend_')
                cascade_legend_handles.append(
                    Line2D(
                        [0], [0], color=c, ls='-', lw=3.0, marker=m, ms=8,
                        markeredgecolor='#222222', markeredgewidth=0.55,
                        label=_cascade_legend_label(name),
                    ),
                )

            ax_c.axhline(1.0, ls='--', color='#555555', lw=1.25)
            _cascade_band_c = _chronic_c
            _span_hi = 3.0
            if _lift_sample:
                _span_hi = max(float(np.nanmax(_lift_sample)) + 0.25, 3.0)
            ax_c.axhspan(1.0, _span_hi, color=_cascade_band_c, alpha=0.08, zorder=0)
            if _lift_sample:
                lo = float(np.nanmin(_lift_sample))
                hi = float(np.nanmax(_lift_sample))
                pad = 0.12 * (hi - lo) if hi > lo else 0.18
                ymin = min(0.5, lo - pad)
                ymin = max(0.0, ymin)
                ymax = max(2.2, hi + pad)
                if ymin < 1.0 < ymax:
                    pass
                elif ymax <= 1.0:
                    ymax = 1.0 + pad
                elif ymin >= 1.0:
                    ymin = max(0.0, 1.0 - pad)
            else:
                ymin, ymax = 0.5, 2.2
            ax_c.set_ylim(ymin, ymax)

            ax_c.set_xlabel(
                f'Year (centre of {cascade_block_years}-yr block; driver lookback = '
                f'{cascade_window_days} step{"s" if cascade_window_days != 1 else ""})',
                fontsize=FIG_AXIS_LABEL_FONTSIZE,
            )
            ax_c.set_ylabel(
                'Cascade lift\n'
                'P(resp. | recent driver) /\n'
                'P(response)',
                fontsize=FIG_AXIS_LABEL_FONTSIZE,
                labelpad=8,
            )
            _c_panel_note = ('true cascade\nsignal', CHRONIC_DIAGNOSTIC_TEXT_GREEN)

        ax_c.tick_params(axis='both', labelsize=_bc_fs)
        # Row 1: solid vs dashed = fixed vs shifting (once). Row 2+: one handle per cascade
        # (color + marker); ncol=2 keeps the baseline key on its own line above cascade names.
        _baseline_legend_handles = [
            Line2D([0], [0], color='#333333', lw=2.75, ls='-', label='Fixed'),
            Line2D(
                [0], [0], color='#333333', lw=2.5, ls='--', dashes=(5, 2.5),
                alpha=0.78, label='Shifting',
            ),
        ]
        _c_leg_handles = _baseline_legend_handles + cascade_legend_handles
        _c_leg_ncol = 2
        _c_leg_fs = 10.5 if len(cascade_legend_handles) >= 3 else 11
        ax_c.legend(
            handles=_c_leg_handles, loc='upper right', frameon=True, framealpha=0.95,
            edgecolor='#bbbbbb', fontsize=_c_leg_fs,
            ncol=_c_leg_ncol, columnspacing=0.9, handlelength=2.0,
        )
        ax_c.grid(True, linestyle=':', linewidth=0.45, color='#bbbbbb', alpha=0.5)
        ax_c.set_axisbelow(True)
        ax_c.set_xlim(yr_min, yr_max)
        _y0, _y1 = ax_c.get_ylim()
        ax_c.text(
            ax_c.get_xlim()[0] + 0.5, _y1 - 0.02 * (_y1 - _y0),
            _c_panel_note[0],
            fontsize=13, color=_c_panel_note[1],
            style='italic', fontweight='semibold', ha='left', va='top',
            zorder=8,
        )
    else:
        ax_c.axis('off')
        ax_c.text(0.5, 0.5,
                  'No cascade_pairs provided.\n'
                  'Pass e.g. {"BHW → Hypoxia": ("BHW", "Hypoxia")}',
                  ha='center', va='center', transform=ax_c.transAxes,
                  fontsize=10, color='#666666', style='italic')

    return fig


# ============================================================================
# WASHINGTON (QUINAULT ROMS) — REAL GCS DATA
# ============================================================================


def _ensure_actea_repo_on_path() -> None:
    """Allow ``from ACTEA_gcs import …`` when executed from StressorAnalysisV2."""
    here = Path(__file__).resolve().parent
    root = here.parent
    for p in (root, here):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _join_monthly_series(raw: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
    """Inner-join several series on calendar month (month-start timestamps)."""
    norm: Dict[str, pd.Series] = {}
    for key, s in raw.items():
        s = s.sort_index()
        idx = pd.to_datetime(s.index).to_period('M').to_timestamp(how='start')
        norm[key] = pd.Series(s.values.astype(float), index=idx, name=key)
    df = pd.DataFrame(norm).dropna()
    return {c: df[c] for c in df.columns}


def _scenario_display_name(scenario: str) -> str:
    s = scenario.lower().replace('-', '')
    return {
        'ssp126': 'SSP1-2.6',
        'ssp245': 'SSP2-4.5',
        'ssp370': 'SSP3-7.0',
        'ssp585': 'SSP5-8.5',
    }.get(s, scenario.upper())


def load_washington_quinault_coastal_timeseries(
    scenario: str = 'ssp245',
    start_year: int = 2017,
    end_year: int = 2060,
) -> Dict[str, object]:
    """Load Washington coast (Quinault ROMS) coastal-mean series from GCS.

    Uses the same file layout and preprocessing as
    ``run_improved_analysis.run_improved_stressor_analysis`` (ACTEA_gcs,
    daily ``tos`` surface file, monthly bottom stats, GLORYS pH, regrid to ROMS
    grid, coastal < 100 m mask from bathymetry).

    Returns
    -------
    dict
        ``sst_daily`` — coastal-mean surface temperature (daily ``pd.Series``).
        ``sst_daily_domain`` — spatial mean daily SST over the full model grid
        (all lat/lon; land/invalid cells omitted via ``skipna``).
        ``monthly`` — inner-joined monthly ``dict`` with keys
        ``sst``, ``bot_temp``, ``o2``, ``ph`` (coastal < 100 m; same as
        ``monthly_by_depth_domain['coastal']``).
        ``monthly_by_depth_domain`` — ``dict`` with keys ``coastal``, ``shelf``,
        ``deep`` (``run_improved_analysis.create_depth_masks``); each value is
        a monthly ``dict`` like ``monthly``.
        ``monthly_domain_mean`` — inner-joined monthly dict (same keys) for the
        **full Quinault grid** spatial mean (all ocean cells; land/invalid via
        ``skipna``), for domain-wide driver figures.
        ``scenario`` — scenario string passed in.
    """
    _ensure_actea_repo_on_path()
    from ACTEA_gcs import ACTEA_gcs
    from run_improved_analysis import (
        _align_mask_to_data,
        _build_file_paths,
        create_depth_masks,
        extract_bathymetry,
    )
    from stressor_analysis_improved import ImprovedStressorAnalysis

    project = 'Quinault_ROMS'
    file_paths = _build_file_paths(project, project, None, scenario)

    gcs_monthly = ACTEA_gcs(frequency='monthly')
    gcs_daily = ACTEA_gcs(frequency='daily')

    base_var_map = {
        'tos_surface': 'tos',
        'thetao_bottom': 'thetao',
        'o2_bottom': 'o2',
        'ph_bottom': 'ph',
    }

    datasets: Dict[str, xr.Dataset] = {}
    data_arrays: Dict[str, xr.DataArray] = {}
    tslice = slice(str(start_year), str(end_year))

    try:
        for var_name, file_path in file_paths.items():
            gcs = gcs_daily if ('daily' in file_path) else gcs_monthly
            print(f'  Loading {var_name} …')
            ds = gcs.open_dataset_on_gs(file_path, decode_times=True)
            if ds is None:
                raise FileNotFoundError(f'GCS object not found: {file_path}')

            datasets[var_name] = ds
            base_var = base_var_map[var_name]
            all_dvars = list(ds.data_vars)
            selected_var = (
                f'{base_var}_mean' if f'{base_var}_mean' in all_dvars
                else base_var if base_var in all_dvars
                else next((v for v in all_dvars if base_var in v), None)
            )
            if selected_var is None:
                raise KeyError(
                    f'{base_var} not in {var_name}; available: {all_dvars}',
                )

            da = ds[selected_var]
            if 'quantile' in da.dims:
                da = da.sel(quantile=0.5, drop=True)
            elif 'quantile' in da.coords:
                da = da.drop_vars('quantile', errors='ignore')

            if 'time' in da.dims:
                da = da.sel(time=tslice)

            data_arrays[var_name] = da
            print(f'    ✓ {selected_var} {dict(da.sizes)}')

        bathymetry = extract_bathymetry(datasets)
        if bathymetry is None:
            raise RuntimeError('Bathymetry not found in datasets')

        depth_masks = create_depth_masks(bathymetry)
        coastal_mask = depth_masks['coastal']

        baseline_period = (2017, 2022)
        analyzer = ImprovedStressorAnalysis(baseline_period=baseline_period)

        if 'tos_surface' not in data_arrays:
            raise KeyError('tos_surface required')

        fine_grid = data_arrays['tos_surface']
        for vn in ('thetao_bottom', 'o2_bottom', 'ph_bottom'):
            if vn in data_arrays:
                data_arrays[vn] = analyzer.regrid_to_fine_resolution(
                    data_arrays[vn], fine_grid, method='linear',
                )

        bathy_lat = bathymetry.sizes.get('lat', bathymetry.sizes.get('latitude', 0))
        fine_lat = fine_grid.sizes.get('lat', fine_grid.sizes.get('latitude', 0))
        if bathy_lat != fine_lat:
            bathymetry = analyzer.regrid_to_fine_resolution(
                bathymetry, fine_grid, method='nearest',
            )
            depth_masks = create_depth_masks(bathymetry)
            coastal_mask = depth_masks['coastal']

        def spatial_mean_masked(
            da: xr.DataArray, mask: xr.DataArray, series_name: str,
        ) -> pd.Series:
            m = _align_mask_to_data(mask, da)
            spatial = [d for d in da.dims if d in ('lat', 'lon', 'latitude', 'longitude')]
            if not spatial:
                raise ValueError(f'No spatial dims in {da.dims}')
            ts = da.where(m).mean(dim=spatial, skipna=True)
            tcoord = pd.DatetimeIndex(pd.to_datetime(ts['time'].values))
            return pd.Series(ts.values.astype(float), index=tcoord, name=series_name)

        def domain_spatial_mean(da: xr.DataArray, series_name: str) -> pd.Series:
            spatial = [d for d in da.dims if d in ('lat', 'lon', 'latitude', 'longitude')]
            if not spatial:
                raise ValueError(f'No spatial dims in {da.dims}')
            ts = da.mean(dim=spatial, skipna=True)
            tcoord = pd.DatetimeIndex(pd.to_datetime(ts['time'].values))
            return pd.Series(ts.values.astype(float), index=tcoord, name=series_name)

        sst_daily = spatial_mean_masked(data_arrays['tos_surface'], coastal_mask, 'SST')
        sst_daily_domain = domain_spatial_mean(data_arrays['tos_surface'], 'SST_domain')

        monthly_by_depth_domain: Dict[str, Dict[str, pd.Series]] = {}
        for dom_name, dom_mask in depth_masks.items():
            sst_dom_daily = spatial_mean_masked(
                data_arrays['tos_surface'], dom_mask, 'SST',
            )
            monthly_raw_d = {
                'sst': sst_dom_daily.resample('MS').mean(),
                'bot_temp': spatial_mean_masked(
                    data_arrays['thetao_bottom'], dom_mask, 'bot_temp',
                ),
                'o2': spatial_mean_masked(data_arrays['o2_bottom'], dom_mask, 'o2'),
                'ph': spatial_mean_masked(data_arrays['ph_bottom'], dom_mask, 'ph'),
            }
            monthly_by_depth_domain[dom_name] = _join_monthly_series(monthly_raw_d)

        monthly = monthly_by_depth_domain['coastal']

        monthly_domain_raw = {
            'sst': sst_daily_domain.resample('MS').mean(),
            'bot_temp': domain_spatial_mean(data_arrays['thetao_bottom'], 'bot_temp'),
            'o2': domain_spatial_mean(data_arrays['o2_bottom'], 'o2'),
            'ph': domain_spatial_mean(data_arrays['ph_bottom'], 'ph'),
        }
        monthly_domain_mean = _join_monthly_series(monthly_domain_raw)

        for ds in datasets.values():
            try:
                ds.close()
            except Exception:
                pass

        out_bundle: Dict[str, object] = {
            'sst_daily': sst_daily,
            'sst_daily_domain': sst_daily_domain,
            'monthly': monthly,
            'monthly_by_depth_domain': monthly_by_depth_domain,
            'monthly_domain_mean': monthly_domain_mean,
            'scenario': scenario,
        }
        return out_bundle
    finally:
        try:
            gcs_monthly.close()
        except Exception:
            pass
        try:
            gcs_daily.close()
        except Exception:
            pass


def _main_washington(
    output_dir: str,
    scenario: str,
    project: str = 'WA_state',
    cascade_window_days: int = 2,
    cascade_block_years: int = 5,
    hypoxia_mode: str = 'percentile',
    spatial_domain: str = 'full',
    exposure_roll_years: int = 5,
    panel_c_mode: str = 'lift',
) -> None:
    """End-to-end pipeline: GCS Quinault Washington bundle → detection → figures."""
    import matplotlib.pyplot as plt

    print('=' * 72)
    print('DUAL-BASELINE STRESSOR ANALYSIS — WASHINGTON (QUINAULT ROMS, GCS)')
    print('=' * 72)
    print(f'Scenario: {scenario}')
    print(f'Spatial domain: {spatial_domain}')
    print(f'Panel (c) mode: {panel_c_mode}')
    if panel_c_mode == 'gridded':
        print(f'  (gridded hypoxia mode: {hypoxia_mode})')

    print('\nLoading Washington series from GCS (ACTEA_gcs)…')
    bundle = load_washington_quinault_coastal_timeseries(scenario=scenario)

    if spatial_domain == 'full':
        mdom = bundle.get('monthly_domain_mean')
        if not isinstance(mdom, dict) or not all(
            k in mdom for k in ('sst', 'bot_temp', 'o2', 'ph')
        ):
            raise RuntimeError(
                'Bundle missing monthly_domain_mean with sst, bot_temp, o2, ph; '
                'reload dual_baseline loader.',
            )
        monthly: Dict[str, pd.Series] = mdom  # type: ignore[assignment]
        sdom = bundle.get('sst_daily_domain')
        if not isinstance(sdom, pd.Series):
            raise RuntimeError(
                'Bundle missing sst_daily_domain for full-domain surface heatwave.',
            )
        sst_daily: pd.Series = sdom  # type: ignore[assignment]
        region_label = 'Washington (Quinault ROMS, full ocean grid mean)'
    else:
        monthly = bundle['monthly']  # type: ignore[assignment]
        sst_daily = bundle['sst_daily']  # type: ignore[assignment]
        region_label = 'Washington coast (coastal < 100 m)'

    if len(monthly) < 4 or not all(k in monthly for k in ('sst', 'bot_temp', 'o2', 'ph')):
        raise RuntimeError(
            f'Expected four aligned monthly series; got keys {list(monthly)}',
        )

    baseline_years = (2017, 2022)
    det_shw_d = DualBaselineDetector(
        baseline_years=baseline_years,
        percentile=90, min_duration=5, direction='above', time_unit='day',
    )
    det_m_above = DualBaselineDetector(
        baseline_years=baseline_years,
        percentile=90, min_duration=3, direction='above', time_unit='month',
    )
    det_m_below = DualBaselineDetector(
        baseline_years=baseline_years,
        percentile=10, min_duration=3, direction='below', time_unit='month',
    )

    print('\nDetecting (daily SHW for Figure 1; monthly for chronicity/cascade)…')
    result_shw_daily = det_shw_d.detect(sst_daily, name='SST')

    # Panel (c) grouped bars need ``panel_c_extras`` + ``panel_c_extras_only`` in
    # ``plot_improved_figure1``. When ``spatial_domain == 'full'``, the daily SHW
    # result *is* domain-mean SST; still attach it here so panel (c) stays the
    # bar chart (fixed vs shifting per year), not the legacy fill_between layout.
    panel_c_extras: List[Tuple[str, DualResult]] = []
    _domain_bar_title = 'Entire model domain (spatial mean SST)'
    if spatial_domain == 'full':
        panel_c_extras.append((_domain_bar_title, result_shw_daily))
    elif spatial_domain == 'coastal' and 'sst_daily_domain' in bundle:
        panel_c_extras.append(
            (
                _domain_bar_title,
                det_shw_d.detect(bundle['sst_daily_domain'], name='SST'),  # type: ignore[arg-type]
            ),
        )

    results_monthly = {
        'SHW':       det_m_above.detect(monthly['sst'],       name='SST'),
        'BHW':       det_m_above.detect(monthly['bot_temp'], name='bot_temp'),
        'Hypoxia':   det_m_below.detect(monthly['o2'],        name='O2'),
        'OA':        det_m_below.detect(monthly['ph'],       name='pH'),
    }
    cascade_pairs = {
        'BHW → Hypoxia': ('BHW', 'Hypoxia'),
        'SHW → Hypoxia': ('SHW', 'Hypoxia'),
        'Hypoxia → OA': ('Hypoxia', 'OA'),
        'BHW → OA': ('BHW', 'OA'),
        'SHW → OA': ('SHW', 'OA'),
    }

    for key, r in results_monthly.items():
        print(
            f"  {key:10s} — "
            f"fixed: n={len(r.fixed.events):4d}  |  "
            f"shifting: n={len(r.shifting.events):4d}",
        )

    print('\nChronic transition years (first 3-yr run with >=50% exposure):')
    for key, r in results_monthly.items():
        exp_f = ChronicityMetrics.annual_exposure(r.fixed.flag)
        exp_s = ChronicityMetrics.annual_exposure(r.shifting.flag)
        y_f = ChronicityMetrics.chronic_transition_year(exp_f)
        y_s = ChronicityMetrics.chronic_transition_year(exp_s)
        print(f'  {key:10s}  fixed: {str(y_f):>6s}  |  shifting: {str(y_s):>6s}')

    scen_label = _scenario_display_name(scenario)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print('\nBuilding figures…')
    fig1 = plot_improved_figure1(
        result_shw_daily,
        region_name=region_label,
        scenario=scen_label,
        percentile=det_shw_d.percentile,
        panel_c_extras=panel_c_extras if panel_c_extras else None,
        panel_c_extras_only=bool(panel_c_extras),
        panel_c_chronicity_palette_ns=len(results_monthly),
        panel_c_chronicity_palette_nc=len(cascade_pairs),
    )
    p1 = out / f'{project}_heatwaves_fixed_shifting_baseline_{scenario}.png'
    fig1.savefig(p1, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig1)
    print(f'  wrote {p1}')

    if panel_c_mode == 'gridded':
        print(
            '\nGridded compound cascading (panel c) — same pipeline as '
            'dual_baseline_sequential_cascade_stats.py …',
        )
        panel_c_gridded = _build_gridded_compound_cascade_bundle(scenario, hypoxia_mode)
    else:
        panel_c_gridded = None

    fig2 = plot_chronicity_and_cascade(
        results_monthly,
        cascade_pairs,
        region_name=region_label,
        cascade_window_days=cascade_window_days,
        cascade_block_years=cascade_block_years,
        exposure_roll_years=exposure_roll_years,
        panel_c_gridded_cascade=panel_c_gridded,
    )
    p2 = out / f'{project}_chronicity_cascade_{scenario}.png'
    fig2.savefig(p2, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    print(f'  wrote {p2}')

    print('\nDone.\n')


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI: real Washington GCS pipeline by default; ``--demo`` for synthetic."""
    parser = argparse.ArgumentParser(
        description='Dual-baseline stressor analysis (Washington GCS or synthetic demo).',
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run synthetic-data demo instead of loading from GCS',
    )
    parser.add_argument(
        '--scenario', '-s',
        default='ssp245',
        help='CMIP6 scenario for GCS paths (default: ssp245)',
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='.',
        help='Directory for PNG outputs',
    )
    parser.add_argument(
        '--project', '-p',
        default='WA_state',
        help='Prefix for output figure filenames (default: WA_state)',
    )
    parser.add_argument(
        '--cascade-block-years',
        type=int,
        default=5,
        metavar='N',
        help=(
            'Panel (c): calendar-year width for CascadeFidelity.lift blocks '
            '(default: 5) and for gridded mode if --panel-c gridded.'
        ),
    )
    parser.add_argument(
        '--cascade-window-steps',
        type=int,
        default=2,
        metavar='N',
        help=(
            'Panel (c) when --panel-c lift: driver look-back in **monthly** time steps '
            '(default: 2 ≈ 2 months). Ignored when --panel-c gridded.'
        ),
    )
    parser.add_argument(
        '--hypoxia-mode',
        choices=('absolute', 'percentile'),
        default='percentile',
        help=(
            'Only with --panel-c gridded: hypoxia extremes — percentile (default) = '
            'monthly O₂ p10; absolute = 1.4 ml/L.'
        ),
    )
    parser.add_argument(
        '--panel-c',
        dest='panel_c_mode',
        choices=('lift', 'gridded'),
        default='lift',
        help=(
            'Panel (c): lift = P(resp|recent driver)/P(resp) on 1D monthly flags (default). '
            'gridded = block mean %% ocean-time cascading (AdvancedStressorMetrics).'
        ),
    )
    parser.add_argument(
        '--spatial-domain',
        choices=('full', 'coastal'),
        default='full',
        help=(
            'WA stressor time series: full = spatial mean over entire ocean grid '
            '(default); coastal = mean over floor <100 m only.'
        ),
    )
    parser.add_argument(
        '--exposure-roll-years',
        type=int,
        default=5,
        metavar='N',
        help='Panel (a): rolling window (years) for annual exposure mean ± 1σ (default: 5).',
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.demo:
        _main_demo(args.output_dir, project=args.project)
    else:
        _main_washington(
            args.output_dir,
            args.scenario,
            project=args.project,
            cascade_window_days=args.cascade_window_steps,
            cascade_block_years=args.cascade_block_years,
            hypoxia_mode=args.hypoxia_mode,
            spatial_domain=args.spatial_domain,
            exposure_roll_years=args.exposure_roll_years,
            panel_c_mode=args.panel_c_mode,
        )


# ============================================================================
# SYNTHETIC-DATA DEMO
# ============================================================================

def _demo_synthetic_data(seed: int = 42) -> Dict[str, pd.Series]:
    """Generate synthetic daily SST, bottom T, bottom O2, and pH for 2017–2060.

    Values are tuned to roughly match the trends reported in QIN v6:
        surface warming 0.032 °C yr-1, bottom warming 0.019 °C yr-1,
        O2 decline -0.5 mmol m-3 yr-1, pH decline -0.0011 units yr-1.
    Returns a dict of pd.Series.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2017-01-01', '2060-12-31', freq='D')
    n = len(dates)
    t = np.arange(n) / 365.25
    doy = dates.dayofyear.values

    # Surface SST
    seasonal = 12.0 + 3.5 * np.cos(2 * np.pi * (doy - 200) / 365.25)
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.88 * ar[i - 1] + rng.normal(0, 0.35)
    sst = seasonal + 0.032 * t + 0.9 * ar \
        + 0.45 * np.sin(2 * np.pi * t / 6.0) \
        + 0.3 * np.sin(2 * np.pi * t / 11.0 + 1.2)
    # Discrete extremes
    for yr, mth, dur, amp in [
        (2029, 11, 60, 0.8), (2044, 5, 100, 1.2), (2047, 8, 90, 1.3),
        (2054, 11, 70, 1.4), (2057, 9, 100, 1.5), (2060, 1, 60, 1.0),
    ]:
        try:
            idx = np.where((dates.year == yr) & (dates.month == mth))[0][0]
            end = min(idx + dur, n)
            xs = np.linspace(-2, 2, end - idx)
            sst[idx:end] += amp * np.exp(-xs ** 2)
        except IndexError:
            pass

    # Bottom T (muted + lagged)
    bot_seasonal = 9.0 + 1.5 * np.cos(2 * np.pi * (doy - 240) / 365.25)
    bot_ar = np.zeros(n)
    for i in range(1, n):
        bot_ar[i] = 0.93 * bot_ar[i - 1] + rng.normal(0, 0.25)
    bot_temp = bot_seasonal + 0.019 * t + 0.7 * bot_ar \
        + 0.12 * (sst - seasonal - 0.032 * t)

    # Bottom O2 (mmol m-3): seasonal minimum in summer, declining trend,
    # negatively correlated with bottom T
    o2_seasonal = 140 - 30 * np.cos(2 * np.pi * (doy - 240) / 365.25)
    o2_ar = np.zeros(n)
    for i in range(1, n):
        o2_ar[i] = 0.94 * o2_ar[i - 1] + rng.normal(0, 6)
    o2 = o2_seasonal - 0.5 * t + o2_ar \
        - 5 * (bot_temp - bot_temp[:365 * 6].mean())

    # pH: seasonal + decline + noise
    ph_seasonal = 7.95 - 0.08 * np.cos(2 * np.pi * (doy - 240) / 365.25)
    ph_ar = np.zeros(n)
    for i in range(1, n):
        ph_ar[i] = 0.93 * ph_ar[i - 1] + rng.normal(0, 0.015)
    ph = ph_seasonal - 0.0011 * t + ph_ar

    make = lambda v, nm: pd.Series(v, index=dates, name=nm)
    return {
        'sst': make(sst, 'sst'),
        'bot_temp': make(bot_temp, 'bot_temp'),
        'o2': make(o2, 'o2'),
        'ph': make(ph, 'ph'),
    }


def _main_demo(
    output_dir: str = '.',
    project: str = 'WA_state',
) -> None:
    """End-to-end demo: synthetic data -> detection -> metrics -> figures."""
    import os

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("DUAL-BASELINE STRESSOR ANALYSIS — SYNTHETIC DEMO")
    print("=" * 72)

    series = _demo_synthetic_data()

    # Configure a detector per stressor
    det_shw = DualBaselineDetector(percentile=90, min_duration=5, direction='above')
    det_bhw = DualBaselineDetector(percentile=90, min_duration=5, direction='above')
    det_hyp = DualBaselineDetector(percentile=10, min_duration=5, direction='below')
    det_oa  = DualBaselineDetector(percentile=10, min_duration=5, direction='below')

    print("\nDetecting events...")
    results = {
        'SHW':       det_shw.detect(series['sst'],      name='SST'),
        'BHW':       det_bhw.detect(series['bot_temp'], name='bot_temp'),
        'Hypoxia':   det_hyp.detect(series['o2'],       name='O2'),
        'OA':        det_oa.detect(series['ph'],        name='pH'),
    }

    for key, r in results.items():
        print(f"  {key:10s} — "
              f"fixed: n={len(r.fixed.events):4d} max_dur={r.fixed.events['duration'].max() if len(r.fixed.events) else 0:4.0f}  |  "
              f"shifting: n={len(r.shifting.events):4d} max_dur={r.shifting.events['duration'].max() if len(r.shifting.events) else 0:4.0f}")

    # ---- Chronicity metrics ----
    print("\nChronic transition years (first 3-yr run with >=50% exposure):")
    for key, r in results.items():
        exp_f = ChronicityMetrics.annual_exposure(r.fixed.flag)
        exp_s = ChronicityMetrics.annual_exposure(r.shifting.flag)
        y_f = ChronicityMetrics.chronic_transition_year(exp_f)
        y_s = ChronicityMetrics.chronic_transition_year(exp_s)
        ci_f = ChronicityMetrics.chronicity_index(r.fixed.flag)
        ci_s = ChronicityMetrics.chronicity_index(r.shifting.flag)
        print(f"  {key:10s}  fixed: {str(y_f):>6s}  (CI={ci_f:.2f})  |  "
              f"shifting: {str(y_s):>6s}  (CI={ci_s:.2f})")

    # ---- Figures ----
    print("\nBuilding figures...")
    fig1 = plot_improved_figure1(
        results['SHW'],
        percentile=det_shw.percentile,
    )
    fig1_path = os.path.join(output_dir, f'{project}_heatwaves_fixed_shifting_baseline_demo.png')
    fig1.savefig(fig1_path, dpi=180, bbox_inches='tight', facecolor='white')
    print(f"  wrote {fig1_path}")

    cascade_pairs = {
        'BHW → Hypoxia': ('BHW', 'Hypoxia'),
        'SHW → Hypoxia': ('SHW', 'Hypoxia'),
        'BHW → OA': ('BHW', 'OA'),
        'SHW → OA': ('SHW', 'OA'),
    }
    fig2 = plot_chronicity_and_cascade(results, cascade_pairs)
    fig2_path = os.path.join(output_dir, f'{project}_chronicity_cascade_demo.png')
    fig2.savefig(fig2_path, dpi=180, bbox_inches='tight', facecolor='white')
    print(f"  wrote {fig2_path}")

    print("\nDone.\n")


if __name__ == '__main__':
    main()
