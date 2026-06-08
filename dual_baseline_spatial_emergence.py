"""
Dual-Baseline Spatial Emergence Analysis
=========================================
Map-based analog of ``plot_chronicity_and_cascade`` from
``dual_baseline_stressor_analysis.py``. Instead of a single coastal-mean
timeseries per stressor, this module runs the dual-baseline detector on every
grid cell and reduces the per-cell flag arrays to a **time of emergence (ToE)**
field — the first year starting a ``run_years``-consecutive run of annual
exposure ≥ ``threshold_pct`` (by default 3 yr of ≥ 50 %).

The three-panel figure reproduces the structure of the 1D cascade figure in
spatial form:

    (a) 4 × 2 small-multiple map grid: ToE per stressor (SHW, BHW, Hypoxia, OA)
        under both the fixed and shifting baselines — direct analog of the
        exposure timelines in the 1D panel (a).
    (b) Two categorical / ordinal maps: (b1) which stressor emerges *first*,
        (b2) the *cascade-complete year* — the year by which ≥ 3 of the 4
        stressors have emerged at each cell — both derived from either the
        shifting or the fixed baseline ToE (two PNGs from the CLI).
        Analog of the transition-year bar chart in the 1D panel (b).
    (c) Driver → response *lag maps*: ``ToE_response − ToE_driver`` in **calendar
        years** (each ToE is a year from annualized exposure; monthly input only
        sets the flags). Same cascade pairs as 1D. Negative: response earlier;
        positive: driver earlier.

Data flow matches ``load_washington_quinault_coastal_timeseries`` in the
sibling module (daily surface T, monthly bottom T, O2, pH regridded to the
fine tos grid), but here the *gridded* arrays are kept and every variable is
resampled to a common **monthly** timestep before running detection. That
keeps detect_gridded memory-tractable and is consistent with the monthly
cascade used in the 1D figure.

Usage
-----
CLI (Washington Quinault ROMS via ``ACTEA_gcs``)::

    PYTHONPATH=/path/to/ACTEA_downscale \\
        python StressorAnalysisV2/dual_baseline_spatial_emergence.py \\
        --scenario ssp245 --output-dir ./outputs --project WA_state

    Outputs:
        ``{project}_spatial_emergence_cascade_{scenario}.png``   (panel a)
        ``{project}_spatial_emergence_b1_b2_shifting_{scenario}.png`` (panel b, shifting)
        ``{project}_spatial_emergence_b1_b2_fixed_{scenario}.png``  (panel b, fixed)
        ``{project}_spatial_emergence_lags_{scenario}.png``      (panel c)

Author: adapted for QIN v6 manuscript
Date: April 2026
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings('ignore', category=RuntimeWarning)

from dual_baseline_stressor_analysis import (  # noqa: E402
    DualBaselineDetector,
    _apply_plot_style,
    chronicity_panel_a_palette,
    chronicity_palette_stressor_rgb,
    dual_baseline_map_extent_lon_lat,
    dual_baseline_year_boundary_cmap_norm,
)

# Stressor ordering shared with the 1D chronicity figure.
STRESSORS: List[str] = ['SHW', 'BHW', 'Hypoxia', 'OA']
STRESSOR_VAR: Dict[str, str] = {
    'SHW': 'sst',
    'BHW': 'bot_temp',
    'Hypoxia': 'o2',
    'OA': 'ph',
}
STRESSOR_DIRECTION: Dict[str, str] = {
    'SHW': 'above',
    'BHW': 'above',
    'Hypoxia': 'below',
    'OA': 'below',
}
STRESSOR_PERCENTILE: Dict[str, float] = {
    'SHW': 90,
    'BHW': 90,
    'Hypoxia': 10,
    'OA': 10,
}

DEFAULT_CASCADE_PAIRS: Dict[str, Tuple[str, str]] = {
    'BHW → Hypoxia': ('BHW', 'Hypoxia'),
    'SHW → Hypoxia': ('SHW', 'Hypoxia'),
    'BHW → OA':      ('BHW', 'OA'),
    'SHW → OA':      ('SHW', 'OA'),
}

BASELINE_YEARS_DEFAULT: Tuple[int, int] = (2017, 2022)
START_YEAR_DEFAULT: int = 2017
END_YEAR_DEFAULT: int = 2060


# ============================================================================
# GRIDDED CHRONIC-TRANSITION YEAR (ToE)
# ============================================================================

def _annual_exposure_gridded(flag: xr.DataArray, time_dim: str = 'time') -> xr.DataArray:
    """Per-cell annual exposure (%) from a monthly/daily boolean flag DataArray.

    Returns an array with dim 'year' instead of the original time dim.
    """
    grp = flag.astype(float).groupby(f'{time_dim}.year').mean(time_dim, skipna=True)
    return grp * 100.0


def chronic_transition_year_grid(
    flag: xr.DataArray,
    threshold_pct: float = 50.0,
    run_years: int = 3,
    time_dim: str = 'time',
) -> xr.DataArray:
    """First year starting a run of `run_years` consecutive yrs with annual
    exposure ≥ threshold_pct, per grid cell.

    Returns a 2-D xr.DataArray (lat, lon) of int years; cells that never
    satisfy the criterion become NaN.
    """
    annual_exp = _annual_exposure_gridded(flag, time_dim=time_dim)
    years = annual_exp['year'].values.astype(int)
    spatial_dims = [d for d in annual_exp.dims if d != 'year']

    # Move 'year' to the front for sliding-window logic.
    arr = annual_exp.transpose('year', *spatial_dims).values  # (ny, *spatial)
    above = arr >= threshold_pct

    ny = above.shape[0]
    if ny < run_years:
        out = np.full(above.shape[1:], np.nan)
    else:
        from numpy.lib.stride_tricks import sliding_window_view

        win = sliding_window_view(above, run_years, axis=0)     # (ny-r+1, *spatial, r)
        all_run = win.all(axis=-1)                              # (ny-r+1, *spatial)
        any_hit = all_run.any(axis=0)
        first_idx = np.argmax(all_run, axis=0)                  # 0 if all False
        toe = np.where(any_hit, years[first_idx], np.nan)
        out = toe

    coords = {d: annual_exp[d] for d in spatial_dims}
    return xr.DataArray(out, dims=spatial_dims, coords=coords, name='toe_year')


# ============================================================================
# DATA LOADING — gridded companion to load_washington_quinault_coastal_timeseries
# ============================================================================

def _ensure_actea_repo_on_path() -> None:
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    for p in (str(repo_root), str(here)):
        if p not in sys.path:
            sys.path.insert(0, p)


def _select_var(ds: xr.Dataset, base_var: str) -> xr.DataArray:
    all_dvars = list(ds.data_vars)
    selected = (
        f'{base_var}_mean' if f'{base_var}_mean' in all_dvars
        else base_var if base_var in all_dvars
        else next((v for v in all_dvars if base_var in v), None)
    )
    if selected is None:
        raise KeyError(f'{base_var} not in dataset; available: {all_dvars}')
    da = ds[selected]
    if 'quantile' in da.dims:
        da = da.sel(quantile=0.5, drop=True)
    elif 'quantile' in da.coords:
        da = da.drop_vars('quantile', errors='ignore')
    return da


def load_gridded_monthly_bundle(
    scenario: str = 'ssp245',
    start_year: int = START_YEAR_DEFAULT,
    end_year: int = END_YEAR_DEFAULT,
) -> Dict[str, object]:
    """Load the four stressor fields as a common monthly-gridded xr.DataArray set.

    Returns a dict with keys:
        monthly    : dict[str, xr.DataArray]  keys 'sst','bot_temp','o2','ph'
                     each with dims (time, lat, lon) on the fine (tos) grid
        bathymetry : xr.DataArray (lat, lon)  ocean-floor depth (m, +down)
        ocean_mask : np.ndarray bool (lat, lon)  True where any variable is finite
        lat, lon   : 1-D coordinate arrays
        scenario   : echo of input
    """
    _ensure_actea_repo_on_path()
    from ACTEA_gcs import ACTEA_gcs  # type: ignore
    from run_improved_analysis import (  # type: ignore
        _build_file_paths,
        create_depth_masks,
        extract_bathymetry,
    )
    from stressor_analysis_improved import ImprovedStressorAnalysis  # type: ignore

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
            da = _select_var(ds, base_var_map[var_name])
            if 'time' in da.dims:
                da = da.sel(time=tslice)
            data_arrays[var_name] = da
            print(f'    ✓ {da.name} {dict(da.sizes)}')

        bathymetry = extract_bathymetry(datasets)
        if bathymetry is None:
            raise RuntimeError('Bathymetry not found in datasets')
        _ = create_depth_masks(bathymetry)  # informational printout

        baseline_period = (2017, 2022)
        analyzer = ImprovedStressorAnalysis(baseline_period=baseline_period)

        if 'tos_surface' not in data_arrays:
            raise KeyError('tos_surface required')
        fine_grid = data_arrays['tos_surface']

        # Regrid bottom variables to the fine tos grid.
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

        # Resample tos daily → monthly so all four share the same time axis.
        sst_monthly = data_arrays['tos_surface'].resample(time='MS').mean(skipna=True)

        def _as_ms(da: xr.DataArray) -> xr.DataArray:
            # Anchor monthly stamps to month-start for clean joins.
            t = pd.DatetimeIndex(pd.to_datetime(da['time'].values)).to_period('M').to_timestamp()
            return da.assign_coords(time=t)

        monthly = {
            'sst':      _as_ms(sst_monthly),
            'bot_temp': _as_ms(data_arrays['thetao_bottom']),
            'o2':       _as_ms(data_arrays['o2_bottom']),
            'ph':       _as_ms(data_arrays['ph_bottom']),
        }

        # Align on the intersection of time axes.
        common_time = pd.DatetimeIndex(monthly['sst']['time'].values)
        for k in ('bot_temp', 'o2', 'ph'):
            common_time = common_time.intersection(
                pd.DatetimeIndex(monthly[k]['time'].values),
            )
        common_time = common_time.sort_values()
        for k in monthly:
            monthly[k] = monthly[k].sel(time=common_time)

        # Ocean mask: any finite value across all variables at t=0.
        finite_stack = np.stack([
            np.isfinite(np.asarray(monthly[k].isel(time=0, drop=True).values, dtype=float))
            for k in ('sst', 'bot_temp', 'o2', 'ph')
        ])
        ocean_mask = finite_stack.all(axis=0)

        # Pull 1-D lat/lon coords (ROMS grid is rectilinear on tos_surface).
        lat_name = 'lat' if 'lat' in fine_grid.dims else 'latitude'
        lon_name = 'lon' if 'lon' in fine_grid.dims else 'longitude'
        lat = np.asarray(fine_grid[lat_name].values, dtype=float)
        lon = np.asarray(fine_grid[lon_name].values, dtype=float)

        for ds in datasets.values():
            try:
                ds.close()
            except Exception:
                pass

        return {
            'monthly': monthly,
            'bathymetry': bathymetry,
            'ocean_mask': ocean_mask,
            'lat': lat,
            'lon': lon,
            'lat_name': lat_name,
            'lon_name': lon_name,
            'scenario': scenario,
        }
    finally:
        try:
            gcs_monthly.close()
        except Exception:
            pass
        try:
            gcs_daily.close()
        except Exception:
            pass


# ============================================================================
# PER-STRESSOR ToE FIELDS
# ============================================================================

def compute_toe_fields(
    monthly: Dict[str, xr.DataArray],
    baseline_years: Tuple[int, int] = BASELINE_YEARS_DEFAULT,
    min_duration_months: int = 3,
    chronic_threshold_pct: float = 50.0,
    chronic_run_years: int = 3,
) -> Dict[str, Dict[str, xr.DataArray]]:
    """Run detect_gridded for each of the 4 stressors and collapse to ToE.

    Returns nested dict::

        {stressor: {'fixed': DataArray(lat,lon),
                    'shifting': DataArray(lat,lon)}}
    """
    toe: Dict[str, Dict[str, xr.DataArray]] = {}
    for stressor in STRESSORS:
        var_key = STRESSOR_VAR[stressor]
        direction = STRESSOR_DIRECTION[stressor]
        pct = STRESSOR_PERCENTILE[stressor]
        data = monthly[var_key]
        print(f'  Detecting {stressor}  (var={var_key}, dir={direction}, p={pct}) …')
        det = DualBaselineDetector(
            baseline_years=baseline_years,
            percentile=pct,
            min_duration=min_duration_months,
            direction=direction,
            time_unit='month',
        )
        ds = det.detect_gridded(data, time_dim='time')
        toe[stressor] = {
            'fixed': chronic_transition_year_grid(
                ds['flag_fixed'], chronic_threshold_pct, chronic_run_years,
            ),
            'shifting': chronic_transition_year_grid(
                ds['flag_shift'], chronic_threshold_pct, chronic_run_years,
            ),
        }
    return toe


# ============================================================================
# DERIVED FIELDS FOR PANEL (b)
# ============================================================================

def first_emerging_stressor(
    toe_by_stressor: Dict[str, xr.DataArray],
    order: Sequence[str] = STRESSORS,
) -> Tuple[xr.DataArray, List[str]]:
    """Per cell, index of the stressor with the smallest ToE.

    Returns (category_index_2d, order_list). Cells where no stressor ever
    emerges are NaN. Ties go to the earliest in ``order``.
    """
    stack = xr.concat(
        [toe_by_stressor[s] for s in order],
        dim=pd.Index(list(order), name='stressor'),
    )
    vals = stack.values  # (n_stressors, lat, lon)
    any_hit = np.isfinite(vals).any(axis=0)
    # argmin on NaN-filled: use np.nanargmin but guard all-NaN cells.
    vals_for_min = np.where(np.isfinite(vals), vals, np.inf)
    idx = np.argmin(vals_for_min, axis=0).astype(float)
    idx[~any_hit] = np.nan
    return xr.DataArray(
        idx, dims=stack.dims[1:], coords={d: stack[d] for d in stack.dims[1:]},
        name='first_emerging_stressor',
    ), list(order)


def cascade_complete_year(
    toe_by_stressor: Dict[str, xr.DataArray],
    order: Sequence[str] = STRESSORS,
    k: int = 3,
) -> xr.DataArray:
    """Per cell, year by which ≥ k of the stressors have emerged.

    Computed as the k-th smallest ToE across the stack. Cells with fewer than
    k emerged stressors in the record are NaN.
    """
    stack = xr.concat(
        [toe_by_stressor[s] for s in order],
        dim=pd.Index(list(order), name='stressor'),
    )
    vals = stack.values
    finite = np.isfinite(vals)
    n_emerged = finite.sum(axis=0)
    sorted_vals = np.sort(np.where(finite, vals, np.inf), axis=0)
    kth = sorted_vals[k - 1]                           # 0-indexed → k-th smallest
    kth = np.where(n_emerged >= k, kth, np.nan)
    return xr.DataArray(
        kth, dims=stack.dims[1:], coords={d: stack[d] for d in stack.dims[1:]},
        name=f'cascade_complete_year_k{k}',
    )


def pair_lag(
    toe_driver: xr.DataArray,
    toe_response: xr.DataArray,
) -> xr.DataArray:
    """Calendar-year lag: ``ToE_response − ToE_driver`` (each ToE is a calendar year).

    Detection uses monthly (or finer) flags, but chronic ToE is defined from **annual**
    exposure and consecutive **calendar** years, so this difference is in years, not
    in month indices.
    """
    return (toe_response - toe_driver).rename('pair_lag_years')


# ============================================================================
# PLOTTING
# ============================================================================

def _diverging_lag_norm(vmax: float):
    import matplotlib.colors as mcolors

    return mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)


def _setup_map_ax(
    ax, lat, lon, ocean_mask, bathymetry=None,
    *,
    draw_left_labels: bool = True,
    draw_bottom_labels: bool = True,
):
    """Cartopy styling; falls back to plain axes if cartopy not installed."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        w, e, s, n = dual_baseline_map_extent_lon_lat(
            np.asarray(lon, dtype=float), np.asarray(lat, dtype=float),
        )
        ax.set_facecolor('white')
        ax.set_xlim(w, e)
        ax.set_ylim(s, n)
        ax.set_aspect('equal', adjustable='datalim')
        ax.tick_params(
            left=draw_left_labels, labelleft=draw_left_labels,
            bottom=draw_bottom_labels, labelbottom=draw_bottom_labels,
        )
        return None

    ax.set_facecolor('white')
    ax.add_feature(
        cfeature.OCEAN, facecolor='white', edgecolor='none', zorder=0,
    )
    # Land below fields (zorder) so coarse NE polygons do not paint over shelf ocean
    # quads; coastlines stay crisp on top.
    ax.add_feature(
        cfeature.LAND, facecolor='#E4E4E4', edgecolor='black',
        linewidth=0.4, zorder=1,
    )
    ax.coastlines(resolution='10m', color='black', linewidth=0.5, zorder=40)
    w, e, s, n = dual_baseline_map_extent_lon_lat(
        np.asarray(lon, dtype=float), np.asarray(lat, dtype=float),
    )
    extent = [w, e, s, n]
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    # Edge lon/lat labels only: meridian/parallel LineCollections from Gridliner
    # can still render above ``pcolormesh`` on some panels (cartopy draw order).
    # Disabling drawn grid lines removes that overlay; tick text still uses the
    # same geometry (see cartopy ``Gridliner._draw_gridliner``).
    gl = ax.gridlines(
        draw_labels={
            'left': draw_left_labels,
            'bottom': draw_bottom_labels,
            'top': False,
            'right': False,
        },
        linewidth=0.3, color='gray',
        alpha=0.4, linestyle='--', zorder=2,
    )
    gl.xlines = False
    gl.ylines = False
    gl.set_zorder(1)
    # Avoid latitude/longitude annotations inside the map (reads as extra columns).
    gl.x_inline = False
    gl.y_inline = False
    gl.left_labels = draw_left_labels
    gl.bottom_labels = draw_bottom_labels
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 7}
    gl.ylabel_style = {'size': 7}
    return ccrs.PlateCarree()


def _plot_field(
    ax, field, lat, lon, cmap, norm, ocean_mask,
    transform=None, never_hatch: bool = True,
):
    """Draw a 2-D field; optional fill for ocean cells that never satisfy ToE/lag."""
    import matplotlib.patches as mpatches  # noqa: F401

    data = np.asarray(field.values if hasattr(field, 'values') else field, dtype=float)
    data = np.where(ocean_mask, data, np.nan)

    # ``edgecolors='face'`` hides QuadMesh cell outlines (often read as a “grid”
    # on top of the field, especially for diverging lag maps).
    if transform is not None:
        im = ax.pcolormesh(
            lon, lat, data, cmap=cmap, norm=norm,
            transform=transform, shading='auto', zorder=25,
            edgecolors='face',
        )
    else:
        im = ax.pcolormesh(
            lon, lat, data, cmap=cmap, norm=norm, shading='auto', zorder=25,
            edgecolors='face',
        )

    if never_hatch:
        never = ocean_mask & ~np.isfinite(data)
        if never.any():
            # Solid neutral fill only: diagonal hatching reads as a dense “grid”
            # on top of the field when many ocean cells are never-emerged / no-lag.
            import matplotlib.colors as mcolors
            hatch_layer = np.where(never, 1.0, np.nan)
            hcmap = mcolors.ListedColormap(['#c8c8c8'])
            _never_kw = dict(
                cmap=hcmap, shading='auto', zorder=26,
                edgecolors='face',
            )
            if transform is not None:
                ax.pcolormesh(
                    lon, lat, hatch_layer, transform=transform, **_never_kw,
                )
            else:
                ax.pcolormesh(lon, lat, hatch_layer, **_never_kw)
    return im


def _resolve_year_range(
    toe: Dict[str, Dict[str, xr.DataArray]],
    year_range: Optional[Tuple[int, int]] = None,
    *,
    baseline: Optional[Literal['fixed', 'shifting']] = None,
) -> Tuple[int, int]:
    """If ``baseline`` is set, only that baseline's ToE values set the default range."""
    bl_keys: Tuple[str, ...] = (
        (baseline,) if baseline is not None else ('fixed', 'shifting')
    )
    all_toe = np.concatenate([
        toe[s][bl].values[np.isfinite(toe[s][bl].values)].astype(float).ravel()
        for s in STRESSORS for bl in bl_keys
    ]) if any(
        np.isfinite(toe[s][bl].values).any()
        for s in STRESSORS for bl in bl_keys
    ) else np.array([])
    if year_range is None:
        if all_toe.size:
            y_lo = int(np.floor(np.nanmin(all_toe) / 5) * 5)
            y_hi = int(np.ceil(np.nanmax(all_toe) / 5) * 5)
            if y_hi == y_lo:
                y_hi = y_lo + 5
        else:
            y_lo, y_hi = 2020, 2060
    else:
        y_lo, y_hi = year_range
    return y_lo, y_hi


def _get_projection():
    try:
        import cartopy.crs as ccrs
        return ccrs.PlateCarree(), True
    except ImportError:
        return None, False


def plot_spatial_emergence_panel_a(
    toe: Dict[str, Dict[str, xr.DataArray]],
    lat: np.ndarray,
    lon: np.ndarray,
    ocean_mask: np.ndarray,
    region_name: str = 'Washington coast (ROMS grid)',
    scenario: str = 'SSP2-4.5',
    year_range: Optional[Tuple[int, int]] = None,
    bathymetry: Optional[xr.DataArray] = None,
    figsize: Tuple[float, float] = (12.4, 14.8),
) -> 'matplotlib.figure.Figure':
    """Panel (a): ToE maps for fixed vs shifting baselines."""
    import matplotlib.pyplot as plt

    _apply_plot_style()
    proj, use_cartopy = _get_projection()
    y_lo, y_hi = _resolve_year_range(toe, year_range=year_range)
    year_cmap, year_norm = dual_baseline_year_boundary_cmap_norm(y_lo, y_hi, n_bins=9)

    fig = plt.figure(figsize=figsize, facecolor='white')
    n_stress = len(STRESSORS)
    # Nested layout: very tight gap between fixed vs shifting columns; separate
    # spacing before the colorbar so the two map columns can sit much closer.
    outer_gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.045], wspace=0.026)
    # Keep wspace >= 0: negative wspace overlaps sibling axes; the second column is
    # drawn on top and can paint over the first (mis-colored strips / “wrong” maps).
    map_gs = outer_gs[0, 0].subgridspec(n_stress, 2, wspace=0.0, hspace=0.042)
    _bl_title = {'fixed': 'Fixed', 'shifting': 'Shifting'}
    for i, stressor in enumerate(STRESSORS):
        for j, bl in enumerate(('fixed', 'shifting')):
            ax = fig.add_subplot(map_gs[i, j], projection=proj) if use_cartopy \
                else fig.add_subplot(map_gs[i, j])
            tf = _setup_map_ax(
                ax, lat, lon, ocean_mask, bathymetry,
                draw_left_labels=(j == 0),
                draw_bottom_labels=(i == n_stress - 1),
            )
            _plot_field(
                ax, toe[stressor][bl], lat, lon,
                year_cmap, year_norm, ocean_mask, transform=tf,
            )
            ax.text(
                0.98, 0.98,
                f'{stressor} — {_bl_title[bl]}',
                transform=ax.transAxes,
                va='top', ha='right', fontsize=9, fontweight='bold',
                bbox={'facecolor': 'white', 'edgecolor': '#CCCCCC', 'alpha': 0.85, 'pad': 1.5},
            )
    cbar_ax = fig.add_subplot(outer_gs[0, 1])
    sm = plt.cm.ScalarMappable(norm=year_norm, cmap=year_cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label('Year of chronic transition (ToE)', fontsize=9)
    cb.ax.tick_params(labelsize=7)

    fig.text(
        0.02, 0.012,
        'chronic = first 3-yr run with >= 50% annual exposure; grey fill = ocean cell that never emerges',
        fontsize=8.5, color='#555555',
    )
    return fig


def plot_spatial_emergence_panel_b(
    toe: Dict[str, Dict[str, xr.DataArray]],
    lat: np.ndarray,
    lon: np.ndarray,
    ocean_mask: np.ndarray,
    region_name: str = 'Washington coast (ROMS grid)',
    scenario: str = 'SSP2-4.5',
    year_range: Optional[Tuple[int, int]] = None,
    bathymetry: Optional[xr.DataArray] = None,
    figsize: Tuple[float, float] = (12.2, 6.0),
    toe_baseline: Literal['fixed', 'shifting'] = 'shifting',
) -> 'matplotlib.figure.Figure':
    """Panel (b): b1 first-emerging stressor + b2 cascade-complete year.

    Both maps use the same dual-baseline ToE fields, selected by ``toe_baseline``
    (``'shifting'`` matches the original manuscript default).
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch

    _apply_plot_style()
    proj, use_cartopy = _get_projection()
    y_lo, y_hi = _resolve_year_range(
        toe, year_range=year_range, baseline=toe_baseline,
    )
    year_cmap, year_norm = dual_baseline_year_boundary_cmap_norm(y_lo, y_hi, n_bins=9)
    toe_sel = {s: toe[s][toe_baseline] for s in STRESSORS}

    fig = plt.figure(figsize=figsize, facecolor='white')
    b_gs = fig.add_gridspec(
        1, 3, width_ratios=[1.0, 1.0, 0.045], wspace=0.06,
    )
    # Same stressor hue order as ``plot_chronicity_and_cascade`` panel (a) / cascade bars.
    _pal_b = chronicity_panel_a_palette(
        len(STRESSORS), len(DEFAULT_CASCADE_PAIRS),
    )
    stress_colors = [
        chronicity_palette_stressor_rgb(i, _pal_b) for i in range(len(STRESSORS))
    ]
    cat_cmap = mcolors.ListedColormap(stress_colors, name='first_emerge')
    cat_bounds = np.arange(-0.5, len(STRESSORS) + 0.5, 1.0)
    cat_norm = mcolors.BoundaryNorm(cat_bounds, cat_cmap.N)

    ax_b1 = fig.add_subplot(b_gs[0, 0], projection=proj) if use_cartopy \
        else fig.add_subplot(b_gs[0, 0])
    tf_b1 = _setup_map_ax(ax_b1, lat, lon, ocean_mask, bathymetry)
    first_idx, order = first_emerging_stressor(toe_sel, order=STRESSORS)
    _plot_field(
        ax_b1, first_idx, lat, lon, cat_cmap, cat_norm, ocean_mask,
        transform=tf_b1,
    )
    handles = [
        Patch(facecolor=stress_colors[i], edgecolor='#333', label=order[i])
        for i in range(len(order))
    ] + [
        Patch(facecolor='#c8c8c8', edgecolor='#333', label='never emerges'),
    ]
    leg_b1 = ax_b1.legend(
        handles=handles,
        loc='upper right',
        bbox_to_anchor=(0.99, 0.99),
        fontsize=7,
        frameon=True,
        facecolor='white',
        framealpha=0.98,
        edgecolor='#222222',
        fancybox=True,
        borderaxespad=0.35,
        ncol=1,
    )
    leg_b1.set_zorder(200)
    if leg_b1.get_frame() is not None:
        leg_b1.get_frame().set_clip_on(False)
        leg_b1.get_frame().set_linewidth(0.6)

    ax_b2 = fig.add_subplot(b_gs[0, 1], projection=proj) if use_cartopy \
        else fig.add_subplot(b_gs[0, 1])
    tf_b2 = _setup_map_ax(
        ax_b2, lat, lon, ocean_mask, bathymetry, draw_left_labels=False,
    )
    cascade_yr = cascade_complete_year(toe_sel, order=STRESSORS, k=3)
    _plot_field(
        ax_b2, cascade_yr, lat, lon, year_cmap, year_norm, ocean_mask,
        transform=tf_b2,
    )
    cbar_ax = fig.add_subplot(b_gs[0, 2])
    # Vertical colorbar: shorten by 30% vs full cax (``shrink`` ignored when ``cax`` set).
    _pos = cbar_ax.get_position()
    _h = _pos.height * 0.7
    _y0 = _pos.y0 + 0.5 * (_pos.height - _h)
    cbar_ax.set_position([_pos.x0, _y0, _pos.width, _h])
    cbar_b = fig.colorbar(
        plt.cm.ScalarMappable(norm=year_norm, cmap=year_cmap),
        cax=cbar_ax,
    )
    cbar_b.set_label('Year', fontsize=8)
    cbar_b.ax.tick_params(labelsize=7)
    return fig


def plot_spatial_emergence_panel_c(
    toe: Dict[str, Dict[str, xr.DataArray]],
    lat: np.ndarray,
    lon: np.ndarray,
    ocean_mask: np.ndarray,
    cascade_pairs: Dict[str, Tuple[str, str]] = DEFAULT_CASCADE_PAIRS,
    region_name: str = 'Washington coast (ROMS grid)',
    scenario: str = 'SSP2-4.5',
    bathymetry: Optional[xr.DataArray] = None,
    figsize: Tuple[float, float] = (15.0, 5.4),
) -> 'matplotlib.figure.Figure':
    """Panel (c): cascade-pair lag maps."""
    import matplotlib.pyplot as plt

    _apply_plot_style()
    proj, use_cartopy = _get_projection()
    fig = plt.figure(figsize=figsize, facecolor='white')

    pair_names = list(cascade_pairs.keys())
    c_gs = fig.add_gridspec(
        1, len(pair_names) + 1, width_ratios=[1.0] * len(pair_names) + [0.045], wspace=0.07,
    )
    pair_fields: Dict[str, xr.DataArray] = {}
    for name, (drv, rsp) in cascade_pairs.items():
        pair_fields[name] = pair_lag(
            toe[drv]['shifting'], toe[rsp]['shifting'],
        )
    pair_arrs = [p.values for p in pair_fields.values()]
    all_lag = np.concatenate([
        a[np.isfinite(a)].ravel() for a in pair_arrs
    ]) if any(np.isfinite(a).any() for a in pair_arrs) else np.array([])
    if all_lag.size:
        vmax = float(np.ceil(np.nanmax(np.abs(all_lag))))
        vmax = max(vmax, 2.0)
    else:
        vmax = 10.0
    lag_norm = _diverging_lag_norm(vmax)
    lag_cmap = plt.get_cmap('RdBu_r')
    for j, name in enumerate(pair_names):
        ax = fig.add_subplot(c_gs[0, j], projection=proj) if use_cartopy \
            else fig.add_subplot(c_gs[0, j])
        tf_c = _setup_map_ax(
            ax, lat, lon, ocean_mask, bathymetry,
            draw_left_labels=(j == 0),
            draw_bottom_labels=True,
        )
        _plot_field(
            ax, pair_fields[name], lat, lon, lag_cmap, lag_norm, ocean_mask,
            transform=tf_c, never_hatch=True,
        )

    cbar_ax = fig.add_subplot(c_gs[0, -1])
    _pos_c = cbar_ax.get_position()
    _h_c = _pos_c.height * 0.7
    _y0_c = _pos_c.y0 + 0.5 * (_pos_c.height - _h_c)
    cbar_ax.set_position([_pos_c.x0, _y0_c, _pos_c.width, _h_c])
    cbar_c = fig.colorbar(
        plt.cm.ScalarMappable(norm=lag_norm, cmap=lag_cmap),
        cax=cbar_ax, orientation='vertical',
    )
    cbar_c.set_label(
        'ΔToE (calendar yr)  ← response earlier    driver earlier →',
        fontsize=8,
    )
    cbar_c.ax.tick_params(labelsize=7)
    return fig


def plot_spatial_emergence_cascade(
    toe: Dict[str, Dict[str, xr.DataArray]],
    lat: np.ndarray,
    lon: np.ndarray,
    ocean_mask: np.ndarray,
    cascade_pairs: Dict[str, Tuple[str, str]] = DEFAULT_CASCADE_PAIRS,
    region_name: str = 'Washington coast (ROMS grid)',
    scenario: str = 'SSP2-4.5',
    year_range: Optional[Tuple[int, int]] = None,
    bathymetry: Optional[xr.DataArray] = None,
    figsize: Tuple[float, float] = (12.4, 14.8),
) -> 'matplotlib.figure.Figure':
    """Backwards-compatible alias that now returns only panel (a)."""
    _ = cascade_pairs  # kept for backward signature compatibility
    return plot_spatial_emergence_panel_a(
        toe=toe,
        lat=lat,
        lon=lon,
        ocean_mask=ocean_mask,
        region_name=region_name,
        scenario=scenario,
        year_range=year_range,
        bathymetry=bathymetry,
        figsize=figsize,
    )


# ============================================================================
# CLI
# ============================================================================

def _scenario_display_name(scenario: str) -> str:
    return {
        'ssp126': 'SSP1-2.6',
        'ssp245': 'SSP2-4.5',
        'ssp370': 'SSP3-7.0',
        'ssp585': 'SSP5-8.5',
    }.get(scenario, scenario.upper())


def _main(output_dir: str, scenario: str, project: str = 'WA_state') -> None:
    import matplotlib.pyplot as plt

    print('=' * 72)
    print('DUAL-BASELINE SPATIAL EMERGENCE — WASHINGTON (QUINAULT ROMS, GCS)')
    print('=' * 72)
    print(f'Scenario: {scenario}')

    print('\nLoading gridded monthly stressor fields from GCS (ACTEA_gcs)…')
    bundle = load_gridded_monthly_bundle(scenario=scenario)
    monthly: Dict[str, xr.DataArray] = bundle['monthly']  # type: ignore[assignment]
    lat = bundle['lat']                                   # type: ignore[assignment]
    lon = bundle['lon']                                   # type: ignore[assignment]
    ocean_mask = bundle['ocean_mask']                     # type: ignore[assignment]
    bathymetry = bundle.get('bathymetry')

    print('\nDetecting and computing ToE per stressor (fixed and shifting)…')
    toe = compute_toe_fields(monthly)

    for s in STRESSORS:
        for bl in ('fixed', 'shifting'):
            arr = toe[s][bl].values
            n_emerged = int(np.isfinite(arr).sum())
            n_ocean = int(ocean_mask.sum())
            print(
                f'  {s:8s} {bl:9s}  emerged: {n_emerged}/{n_ocean} cells  '
                f'median yr: {np.nanmedian(arr):.0f}' if n_emerged > 0
                else f'  {s:8s} {bl:9s}  emerged: 0/{n_ocean} cells',
            )

    scen_label = _scenario_display_name(scenario)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print('\nBuilding panel (a): fixed vs shifting ToE maps…')
    fig_a = plot_spatial_emergence_panel_a(
        toe, lat=lat, lon=lon, ocean_mask=ocean_mask,
        region_name='Washington coast (Quinault ROMS grid)',
        scenario=scen_label,
        bathymetry=bathymetry,
    )
    out_a = out / f'{project}_spatial_emergence_cascade_{scenario}.png'
    fig_a.savefig(out_a, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig_a)
    print(f'  wrote {out_a}')

    print('\nBuilding panel (b): b1/b2 cascade summary maps (shifting baseline)…')
    fig_b = plot_spatial_emergence_panel_b(
        toe, lat=lat, lon=lon, ocean_mask=ocean_mask,
        region_name='Washington coast (Quinault ROMS grid)',
        scenario=scen_label,
        bathymetry=bathymetry,
        toe_baseline='shifting',
    )
    out_b = out / f'{project}_spatial_emergence_b1_b2_shifting_{scenario}.png'
    fig_b.savefig(out_b, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig_b)
    print(f'  wrote {out_b}')

    print('\nBuilding panel (b): b1/b2 cascade summary maps (fixed baseline)…')
    fig_b_fix = plot_spatial_emergence_panel_b(
        toe, lat=lat, lon=lon, ocean_mask=ocean_mask,
        region_name='Washington coast (Quinault ROMS grid)',
        scenario=scen_label,
        bathymetry=bathymetry,
        toe_baseline='fixed',
    )
    out_b_fix = out / f'{project}_spatial_emergence_b1_b2_fixed_{scenario}.png'
    fig_b_fix.savefig(out_b_fix, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig_b_fix)
    print(f'  wrote {out_b_fix}')

    print('\nBuilding panel (c): cascade lag maps…')
    fig_c = plot_spatial_emergence_panel_c(
        toe, lat=lat, lon=lon, ocean_mask=ocean_mask,
        region_name='Washington coast (Quinault ROMS grid)',
        scenario=scen_label,
        bathymetry=bathymetry,
    )
    out_c = out / f'{project}_spatial_emergence_lags_{scenario}.png'
    fig_c.savefig(out_c, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig_c)
    print(f'  wrote {out_c}')
    print('\nDone.\n')


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description='Dual-baseline spatial emergence cascade (GCS pipeline).',
    )
    parser.add_argument(
        '--scenario', '-s', default='ssp245',
        help='CMIP6 scenario for GCS paths (default: ssp245)',
    )
    parser.add_argument(
        '--output-dir', '-o', default='.',
        help='Directory for PNG outputs',
    )
    parser.add_argument(
        '--project', '-p', default='WA_state',
        help='Prefix for output figure filename (default: WA_state)',
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    _main(args.output_dir, args.scenario, project=args.project)


if __name__ == '__main__':
    main()