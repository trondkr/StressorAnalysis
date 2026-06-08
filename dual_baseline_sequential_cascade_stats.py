#!/usr/bin/env python3
"""
Recompute sequential / cascading compound statistics under fixed vs shifting baselines.

Context
-------
``run_improved_analysis.py`` (and ``run_analysis_actea.sh``) use
``ImprovedStressorAnalysis`` with a **fixed** 2017–2022 climatology for
percentile-based heatwaves and acidification, plus a **fixed** hypoxia
concentration threshold (1.4 ml/L). No shifting-baseline detection is applied
there.

This script builds monthly extreme-event datasets for **both**:

* **Fixed baseline** — same percentile logic as ``DualBaselineDetector`` /
  ``dual_baseline_spatial_emergence`` (threshold from 2017–2022 monthly
  climatology, constant in calendar month).
* **Shifting baseline** — threshold follows the linear trend offset used in
  ``DualBaselineDetector.detect_gridded`` (Amaya et al. 2023-style).

Hypoxia can be detected either as **absolute** (1.4 ml/L, matching the
streamlined pipeline) or **percentile** (10th %ile of O₂, matching the dual
baseline spatial figures). Default: **absolute** for parity with
``run_improved_analysis``.

Sequential / cascading definitions follow ``AdvancedStressorMetrics`` with
``temporal_window=1``, ``lag_threshold=1``, ``time_unit='month'`` (one-month
lag, comparable to the ~30-day narrative when working from monthly means).

**Not the same as** ``WA_state_chronicity_cascade_*.png`` **panel (c)** (lower right):

* **This script** reports **domain-mean % of months** that are ``sequential_*`` or
  ``cascading_*`` from ``classify_compound_events`` on **full-grid** monthly fields
  (ocean-masked mean over lat/lon). Cascading requires response **intensity** to
  exceed driver intensity at ≤1-month lag (see ``advanced_stressor_metrics``).
* **Panel (c)** plots **CascadeFidelity.lift** = P(response | driver in recent
  window) / P(response), from **coastal (<100 m) mean** monthly flags, in
  ``cascade_block_years``-year blocks (default 5 yr) with a short driver look-back
  (default 2 **monthly** steps). Lift can rank pairs differently than the
  intensity-based cascade % above.
* **Hypoxia** in the WA chronicity figure uses **monthly p10** below baseline
  (``DualBaselineDetector`` on coastal O₂). This script’s default ``absolute``
  hypoxia (1.4 ml/L) matches ``run_improved_analysis`` style, **not** that figure —
  use ``--hypoxia-mode percentile`` for the same hypoxia definition as the PNG’s
  Hypoxia mask (still **grid** coastal mean ≠ identical to 1D coastal series, but
  much closer).

Usage
-----
    PYTHONPATH=/path/to/ACTEA_downscale \\
        python StressorAnalysisV2/dual_baseline_sequential_cascade_stats.py \\
            --scenario ssp245

Requires GCS access to the same Quinault ROMS bundles as other StressorAnalysisV2 tools.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple, cast

import numpy as np
import xarray as xr

# -----------------------------------------------------------------------------
# Imports (StressorAnalysisV2 as script cwd)
# -----------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from dual_baseline_spatial_emergence import load_gridded_monthly_bundle  # noqa: E402
from stressor_analysis_improved import ImprovedStressorAnalysis  # noqa: E402
from advanced_stressor_metrics import AdvancedStressorMetrics  # noqa: E402

BaselineMode = Literal['fixed', 'shifting']
HypoxiaMode = Literal['absolute', 'percentile']

BASELINE_YEARS: Tuple[int, int] = (2017, 2022)

# Keys expected by AdvancedStressorMetrics sequential/cascading pairs
STRESSOR_SPECS = [
    ('surface_heatwave', 'sst', 'above', 90, 5),
    ('bottom_heatwave', 'bot_temp', 'above', 90, 3),
    ('acidification', 'ph', 'below', 10, 3),
]


def _apply_duration(da: xr.DataArray, min_duration: int) -> xr.DataArray:
    """Trailing-window minimum-duration filter (matches ImprovedStressorAnalysis)."""
    roll = da.astype(float).rolling(time=min_duration, center=False).sum()
    sustained = roll >= min_duration
    return (da.astype(bool) & sustained).astype(bool)


def _absolute_hypoxia_extremes(
    o2: xr.DataArray,
    threshold_ml_l: float,
    min_duration: int,
    ocean_mask: np.ndarray,
    analyzer: ImprovedStressorAnalysis,
) -> xr.Dataset:
    """Fixed absolute hypoxia threshold (no shifting baseline variant)."""
    o2m = analyzer.apply_land_mask(o2, _ocean_mask_da(o2, ocean_mask))
    thr = xr.full_like(o2m, float(threshold_ml_l))
    exceeds = o2m < thr
    is_extreme = analyzer._apply_duration_filter(exceeds, min_duration, max_gap=2)
    intensity = (thr - o2m).where(exceeds, 0.0)
    intensity = intensity.where(is_extreme, 0.0)
    return xr.Dataset({
        'is_extreme': is_extreme,
        'intensity': intensity,
        'threshold': thr,
    })


def _ocean_mask_da(da: xr.DataArray, ocean_mask: np.ndarray) -> xr.DataArray:
    lat_n = 'lat' if 'lat' in da.dims else 'latitude'
    lon_n = 'lon' if 'lon' in da.dims else 'longitude'
    return xr.DataArray(
        ocean_mask,
        dims=(lat_n, lon_n),
        coords={lat_n: da[lat_n], lon_n: da[lon_n]},
    )


def _numpy_slope_per_cell(da: xr.DataArray, t_years: np.ndarray) -> xr.DataArray:
    """Linear regression slope of ``da`` vs ``t_years`` along ``time`` (robust vs cftime)."""
    v = np.asarray(da.values, dtype=np.float64)
    if v.ndim == 0:
        return xr.DataArray(0.0)
    t = np.asarray(t_years, dtype=np.float64).reshape(-1)
    T = v.shape[0]
    if len(t) != T:
        raise ValueError('t_years length must match time size')
    spatial_dims = [d for d in da.dims if d != 'time']
    v2 = v.reshape(T, -1)
    tm = t - np.mean(t)
    den = float(np.dot(tm, tm))
    if den <= 0:
        slope_flat = np.zeros(v2.shape[1], dtype=np.float64)
    else:
        vv = v2 - np.nanmean(v2, axis=0, keepdims=True)
        num = np.nansum(tm[:, np.newaxis] * vv, axis=0)
        slope_flat = num / den
    slope_arr = slope_flat.reshape(v.shape[1:])
    coords = {d: da[d] for d in spatial_dims}
    return xr.DataArray(slope_arr, dims=spatial_dims, coords=coords)


def _vectorized_monthly_thresholds(
    da: xr.DataArray,
    baseline_years: Tuple[int, int],
    percentile: float,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Fixed vs shifting monthly thresholds (same construction as ``DualBaselineDetector``).

    Fixed: calendar-month quantile over baseline years, expanded to each time step.
    Shifting: fixed threshold plus per-cell linear trend in time, centred so the
    baseline-period mean offset is zero (Amaya et al. 2023-style).
    """
    q = percentile / 100.0
    y0, y1 = baseline_years
    sub = da.sel(time=slice(f'{y0}', f'{y1}'))
    thr_by_month = sub.groupby('time.month').quantile(q, dim='time', skipna=True)
    cal_month = da['time'].dt.month
    thr_f = thr_by_month.sel(month=cal_month)
    thr_f = thr_f.drop_vars('month', errors='ignore')

    t_ns = da['time'].values.astype('datetime64[ns]')
    t_delta_days = (t_ns - t_ns[0]).astype('timedelta64[D]').astype(float) / 365.25
    t_years = np.asarray(t_delta_days, dtype=np.float64)
    years = da['time'].dt.year.values
    bm = (years >= y0) & (years <= y1)
    tb_mean = float(np.mean(t_years[bm])) if np.any(bm) else float(np.mean(t_years))

    slope = _numpy_slope_per_cell(da, t_years)
    slope = slope.where(np.isfinite(slope), other=0.0)
    trend_off = slope * (xr.DataArray(t_years, dims=['time'], coords={'time': da.time}) - tb_mean)
    thr_s = thr_f + trend_off
    return thr_f, thr_s


def _percentile_extremes_from_dual(
    da: xr.DataArray,
    direction: str,
    percentile: float,
    min_duration: int,
    mode: BaselineMode,
    ocean_mask: np.ndarray,
    analyzer: ImprovedStressorAnalysis,
) -> xr.Dataset:
    """Percentile exceedance (vectorized thresholds), then minimum-duration filter."""
    da_m = analyzer.apply_land_mask(da, _ocean_mask_da(da, ocean_mask))
    thr_f, thr_s = _vectorized_monthly_thresholds(da_m, BASELINE_YEARS, percentile)
    thresh = thr_f if mode == 'fixed' else thr_s
    if direction == 'above':
        exceeds = da_m > thresh
        intensity_raw = (da_m - thresh).where(exceeds, 0.0)
    else:
        exceeds = da_m < thresh
        intensity_raw = (thresh - da_m).where(exceeds, 0.0)
    is_extreme = _apply_duration(exceeds, min_duration)
    intensity = intensity_raw.where(is_extreme, 0.0)
    return xr.Dataset({
        'is_extreme': is_extreme,
        'intensity': intensity,
        'threshold': thresh,
    })


def build_extremes_monthly(
    monthly: Dict[str, xr.DataArray],
    ocean_mask: np.ndarray,
    hypoxia_mode: HypoxiaMode,
    mode: BaselineMode,
) -> Dict[str, xr.Dataset]:
    analyzer = ImprovedStressorAnalysis(baseline_period=BASELINE_YEARS)
    out: Dict[str, xr.Dataset] = {}
    for key, var, direction, pct, min_d in STRESSOR_SPECS:
        out[key] = _percentile_extremes_from_dual(
            monthly[var], direction, pct, min_d, mode, ocean_mask, analyzer,
        )
    o2 = monthly['o2']
    if hypoxia_mode == 'absolute':
        out['hypoxia'] = _absolute_hypoxia_extremes(
            o2, 1.4, 3, ocean_mask, analyzer,
        )
    else:
        out['hypoxia'] = _percentile_extremes_from_dual(
            o2, 'below', 10, 3, mode, ocean_mask, analyzer,
        )
    return out


def domain_mean_frequency(
    da: xr.DataArray,
    ocean_mask: np.ndarray,
) -> float:
    """Mean probability (0–100 %) over ocean cells and full time range."""
    m = xr.DataArray(ocean_mask, dims=da.dims[-2:], coords={d: da.coords[d] for d in da.dims[-2:]})
    masked = da.where(m)
    return float(masked.mean(skipna=True)) * 100.0


def shelf_band_means(
    bathy: xr.DataArray,
    seq: xr.DataArray,
    ocean_mask: np.ndarray,
    coastal_max_m: float = 100.0,
    offshore_min_m: float = 250.0,
) -> Tuple[float, float]:
    """Rough nearshore vs offshore mean sequential frequency (%)."""
    lat_name = 'lat' if 'lat' in bathy.dims else 'latitude'
    lon_name = 'lon' if 'lon' in bathy.dims else 'longitude'
    bathy = bathy.rename({lat_name: 'lat', lon_name: 'lon'})
    near = (bathy <= coastal_max_m) & ocean_mask
    off = (bathy >= offshore_min_m) & ocean_mask
    m_near = xr.DataArray(near, dims=seq.dims[-2:], coords={'lat': seq['lat'], 'lon': seq['lon']})
    m_off = xr.DataArray(off, dims=seq.dims[-2:], coords={'lat': seq['lat'], 'lon': seq['lon']})
    fn = float(seq.where(m_near).mean(skipna=True)) * 100.0
    fo = float(seq.where(m_off).mean(skipna=True)) * 100.0
    return fn, fo


def run_stats(
    scenario: str,
    hypoxia_mode: HypoxiaMode,
) -> Dict[str, object]:
    bundle = cast(Any, load_gridded_monthly_bundle(scenario=scenario))
    monthly = {
        k: (v.compute() if hasattr(v.data, 'compute') else v)
        for k, v in cast(Dict[str, xr.DataArray], bundle['monthly']).items()
    }
    ocean_mask = cast(np.ndarray, bundle['ocean_mask'])
    bathy = cast(xr.DataArray, bundle['bathymetry'])

    metrics = AdvancedStressorMetrics(baseline_period=BASELINE_YEARS)
    results: Dict[str, Dict[str, float]] = {'fixed': {}, 'shifting': {}}

    for mode in ('fixed', 'shifting'):
        extremes = build_extremes_monthly(monthly, ocean_mask, hypoxia_mode, mode)  # type: ignore[arg-type]
        compound = metrics.classify_compound_events(
            extremes,
            temporal_window=1,
            lag_threshold=1,
            time_unit='month',
        )
        for k, da in compound.items():
            if not (k.startswith('sequential_') or k.startswith('cascading_')):
                continue
            if 'time' not in da.dims:
                continue
            results[mode][k] = domain_mean_frequency(da.astype(float), ocean_mask)

        # Hypoxia → OA cross-shelf (sequential only) when both exist
        hk = 'sequential_hypoxia_to_acidification'
        if hk in compound:
            fn, fo = shelf_band_means(bathy, compound[hk].astype(float), ocean_mask)
            results[mode][f'{hk}_nearshore_pct_mean'] = fn
            results[mode][f'{hk}_offshore_pct_mean'] = fo

    return {
        'scenario': scenario,
        'hypoxia_mode': hypoxia_mode,
        'baseline_years': list(BASELINE_YEARS),
        'domain_mean_pct': results,
    }


def _pair_pct(res: Dict[str, Dict[str, float]], mode: str, prefix: str, a: str, b: str) -> float:
    k = f'{prefix}_{a}_to_{b}' if prefix == 'sequential' else f'{prefix}_{a}_triggers_{b}'
    return res[mode].get(k, float('nan'))


def format_summary_lines(stats: Dict[str, object]) -> List[str]:
    """Human-readable bullets mirroring manuscript section 3.4 structure."""
    r: Dict[str, Dict[str, float]] = stats['domain_mean_pct']  # type: ignore[assignment]
    hyp = stats['hypoxia_mode']
    scen = stats['scenario']

    def lines_for(mode: str) -> List[str]:
        seq = lambda x, y: _pair_pct(r, mode, 'sequential', x, y)  # noqa: E731
        cas = lambda x, y: _pair_pct(r, mode, 'cascading', x, y)  # noqa: E731

        six = [
            seq('surface_heatwave', 'bottom_heatwave'),
            seq('surface_heatwave', 'hypoxia'),
            seq('surface_heatwave', 'acidification'),
            seq('bottom_heatwave', 'hypoxia'),
            seq('bottom_heatwave', 'acidification'),
            seq('hypoxia', 'acidification'),
        ]
        six_valid = [v for v in six if np.isfinite(v)]
        rng = (min(six_valid), max(six_valid)) if six_valid else (float('nan'), float('nan'))

        near = r[mode].get('sequential_hypoxia_to_acidification_nearshore_pct_mean', float('nan'))
        off = r[mode].get('sequential_hypoxia_to_acidification_offshore_pct_mean', float('nan'))

        bhw_hyp_seq = seq('bottom_heatwave', 'hypoxia')
        bhw_hyp_cas = cas('bottom_heatwave', 'hypoxia')
        shw_hyp_seq = seq('surface_heatwave', 'hypoxia')
        shw_hyp_cas = cas('surface_heatwave', 'hypoxia')
        frac_shw = (shw_hyp_cas / shw_hyp_seq * 100) if shw_hyp_seq > 0 else float('nan')

        oa_cascades = [
            cas('surface_heatwave', 'acidification'),
            cas('bottom_heatwave', 'acidification'),
            cas('hypoxia', 'acidification'),
        ]
        oa_seq = [
            seq('surface_heatwave', 'acidification'),
            seq('bottom_heatwave', 'acidification'),
            seq('hypoxia', 'acidification'),
        ]
        oa_c_mean = float(np.nanmean(oa_cascades))
        oa_s_mean = float(np.nanmean(oa_seq))
        frac_oa = (oa_c_mean / oa_s_mean * 100) if oa_s_mean > 0 else float('nan')

        frac_bhw = (bhw_hyp_cas / bhw_hyp_seq * 100) if bhw_hyp_seq > 0 else float('nan')

        return [
            f"Scenario {scen}; hypoxia = {hyp}; baseline mode = **{mode.upper()}** (threshold definition).",
            "",
            "**Sequential events** (one stressor within one month of another; domain-mean % of months):",
            f"* Six canonical pairs — range **{rng[0]:.1f}–{rng[1]:.1f}%** across pairs.",
            f"* BHW → OA: **{seq('bottom_heatwave', 'acidification'):.1f}%**; Hypoxia → OA: **{seq('hypoxia', 'acidification'):.1f}%**.",
            f"* Hypoxia → OA cross-shelf (sequential): nearshore (~≤100 m) **{near:.1f}%**, offshore (≥250 m) **{off:.1f}%**.",
            f"* BHW → Hypoxia: **{bhw_hyp_seq:.1f}%**.",
            "",
            "**Cascading** (≤1 month lag & response intensity exceeds driver intensity; domain-mean % of months):",
            f"* BHW → Hypoxia: **{bhw_hyp_cas:.1f}%**.",
            f"* SHW → Hypoxia: **{shw_hyp_cas:.1f}%**.",
            f"* SHW → BHW: **{cas('surface_heatwave', 'bottom_heatwave'):.1f}%**.",
            f"* OA-related cascades — SHW → OA **{oa_cascades[0]:.1f}%**, BHW → OA **{oa_cascades[1]:.1f}%**, Hypoxia → OA **{oa_cascades[2]:.1f}%** (mean **{oa_c_mean:.2f}%**).",
            "",
            "**Sequential vs cascading interpretation:**",
            f"* BHW–Hypoxia: sequential **{bhw_hyp_seq:.1f}%**, cascading **{bhw_hyp_cas:.1f}%** (~**{frac_bhw:.0f}%** of sequential months also satisfy the cascade intensity criterion).",
            f"* SHW–Hypoxia: sequential **{shw_hyp_seq:.1f}%**, cascading **{shw_hyp_cas:.1f}%** (~**{frac_shw:.0f}%** of sequential months are cascades).",
            f"* Acidification pathways: mean cascading **{oa_c_mean:.2f}%** vs mean sequential **{oa_s_mean:.2f}%** (~**{frac_oa:.1f}%** cascading fraction of sequential co-occurrences).",
        ]

    hyp_note = (
        "hypoxia = absolute 1.4 ml/L (same under fixed vs shifting because it is not percentile-based)."
        if hyp == 'absolute'
        else "hypoxia = monthly O₂ p10 (threshold shifts with the trend like other percentile stressors)."
    )
    out = [
        "### Section 3.4 (recalculated): Sequential and cascading stressor relationships",
        "",
        "_Detection: Quinault ROMS SSP bundle on a common monthly grid; SST/bottom T/pH use 2017–2022 "
        "monthly climatology percentiles (p90 warm, p10 acidification); "
        f"{hyp_note} "
        "Shifting baseline adds a per-cell linear trend to that monthly threshold (``DualBaselineDetector`` logic). "
        "Sequential window = 1 month; cascade = ≤1-month lag with higher response intensity._",
        "",
    ]
    out.extend(lines_for('fixed'))
    out.extend(["", "---", ""])
    out.extend(lines_for('shifting'))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scenario', default='ssp245')
    p.add_argument(
        '--hypoxia-mode', choices=('absolute', 'percentile'), default='absolute',
        help=(
            'absolute: fixed 1.4 ml/L (matches streamlined run_improved_analysis). '
            'percentile: monthly O₂ below p10 with fixed/shifting thresholds — closer '
            'to Hypoxia in WA_state_chronicity_cascade_*.png (still full-grid mean here).'
        ),
    )
    p.add_argument('--output-json', type=Path, default=None, help='Write full numeric dict to this path')
    args = p.parse_args()

    print(
        '\n'.join([
            '',
            '—' * 72,
            'Relation to WA_state_chronicity_cascade_*.png (panel c, lower right):',
            '  • PNG = CascadeFidelity LIFT on coastal-mean monthly FLAGS (blocks & lookback',
            '    set in dual_baseline_stressor_analysis.py / CLI --cascade-*).',
            '  • This script = AdvancedStressorMetrics domain-mean % on FULL GRID.',
            '  • For Hypoxia comparable to that PNG, re-run with --hypoxia-mode percentile.',
            '—' * 72,
            '',
        ]),
    )
    stats = run_stats(args.scenario, args.hypoxia_mode)  # type: ignore[arg-type]
    lines = format_summary_lines(stats)
    print('\n'.join(lines))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(stats, f, indent=2, default=float)


if __name__ == '__main__':
    main()
