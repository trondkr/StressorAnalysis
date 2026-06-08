#!/usr/bin/env python3
"""High-resolution shallow bathymetry map for Washington (ETOPO1 via ``ACTEA_bathymetry``).

Uses the same standalone pattern as Kodiak / GOA egg analysis: ``Bathymetry()`` plus
``add_bathymetry_from_etopo1`` with ``focus_upper_depth_m`` so contour levels pack into
the upper water column. Colormap: **Crameri Oslo** (`crameri:oslo`) with ends **trimmed** so depth shading does not collapse to near-black / near-white across the whole map.

Also writes a **standalone regional map** (the same extent as the main figure’s inset: 130–120°W, 42–52°N) using **ColorBrewer Greys** nine-step via ``cmap`` (``colorbrewer:greys_9`` / ``Greys_9``). Disable with ``--skip-regional-greys`` or set path with ``--regional-greys-output``.

Example::

    PYTHONPATH=/path/to/ACTEA_downscale \\
        python StressorAnalysisV2/plot_washington_bathymetry_etopo1.py \\
        --output ./StressorAnalysisV2/WA_state_bathymetry_etopo1_ylgnbu.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ACTEA_bathymetry import Bathymetry  # noqa: E402

# Regional inset extent (Pacific NW shelf / slope), reused for standalone Greys map.
INSET_LON_W, INSET_LON_E = -130.0, -120.0
INSET_LAT_S, INSET_LAT_N = 42.0, 52.0
INSET_FOCUS_UPPER_DEPTH_M = 1200.0


def _truncate_cmap_display(
    cmap,
    lo: float = 0.17,
    hi: float = 0.90,
    n: int = 256,
):
    """Use only the middle of ``cmap`` (Oslo-like ramps) so extremes do not dominate the figure."""
    lo = float(np.clip(lo, 0.0, 0.45))
    hi = float(np.clip(hi, 0.55, 1.0))
    if hi <= lo + 1e-3:
        hi = lo + 0.5
    rgba = cmap(np.linspace(lo, hi, n))
    rgb = [tuple(float(rgba[i, j]) for j in range(3)) for i in range(n)]
    name = f"{getattr(cmap, 'name', 'cmap')}_bathy_display"
    out = mcolors.LinearSegmentedColormap.from_list(name, rgb, N=n)
    # Softer than Oslo's floor for extend='min' abyssal fill
    und = cmap(float(lo) * 0.55)
    out.set_under(tuple(float(und[i]) for i in range(3)))
    return out


def _load_oslo_mpl():
    """Crameri *oslo* via ``cmap`` (``crameri:oslo``), trimmed for bathymetry display."""
    try:
        from cmap import Colormap

        base = None
        for name in (
            "crameri:oslo",
            "oslo",
            "matplotlib:oslo",
        ):
            try:
                base = Colormap(name).to_mpl()
                break
            except Exception:
                continue
        if base is None:
            base = plt.get_cmap("viridis")
        return _truncate_cmap_display(base)
    except Exception:
        return _truncate_cmap_display(plt.get_cmap("viridis"))


def _load_greys_9_mpl():
    """ColorBrewer Greys (9 classes) via ``cmap``; user-facing name often written *greys_09*."""
    try:
        from cmap import Colormap

        for name in (
            "colorbrewer:greys_9",
            "colorbrewer:Greys_9",
            "colorbrewer:greys_09",
            "Greys_9",
        ):
            try:
                return Colormap(name).to_mpl()
            except Exception:
                continue
    except Exception:
        pass
    return plt.get_cmap("Greys")


def plot_washington_bathymetry_regional_inset_standalone(
    output_path: Path,
    *,
    bathy_alpha: float = 0.95,
    dpi: int = 220,
    focus_upper_depth_m: float = INSET_FOCUS_UPPER_DEPTH_M,
) -> Path:
    """Full-page map for the regional inset extent only (Greys colormap, ETOPO1)."""
    cmap_mpl = _load_greys_9_mpl()
    fig = plt.figure(figsize=(10.5, 8.4), facecolor="white")
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(
        [INSET_LON_W, INSET_LON_E, INSET_LAT_S, INSET_LAT_N],
        crs=ccrs.PlateCarree(),
    )

    bath = Bathymetry()
    cf = bath.add_bathymetry_from_etopo1(
        ax=ax,
        ranges=[INSET_LON_W, INSET_LON_E, INSET_LAT_S, INSET_LAT_N],
        cmap=cmap_mpl,
        alpha=bathy_alpha,
        zorder=0,
        lon_plot_negative180=True,
        focus_upper_depth_m=focus_upper_depth_m,
        draw_contour_lines=False,
        extend="min",
    )

    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#d4d2ce", zorder=1)
    ax.add_feature(
        cfeature.COASTLINE.with_scale("10m"),
        linewidth=0.55,
        edgecolor="#1a1a1a",
        zorder=3,
    )
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.35, edgecolor="#666666", zorder=2)

    gl = ax.gridlines(draw_labels=True, linewidth=0.35, alpha=0.55, linestyle="--", color="#555555")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 11}
    gl.ylabel_style = {"size": 11}

    cb = fig.colorbar(cf, ax=ax, fraction=0.035, pad=0.02, extend="min")
    cb.set_label(
        f"Depth (m) · contour levels 0–{focus_upper_depth_m:.0f} m\n"
        "(deeper offshore → under-color; ETOPO1 · ColorBrewer Greys 9)",
        fontsize=10,
    )

    fig.suptitle(
        "Pacific NW regional bathymetry — ETOPO1\n"
        "130–120°W, 42–52°N · colorbrewer:greys_9 (cmap)",
        fontsize=13,
        fontweight="semibold",
        y=0.98,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_washington_bathymetry(
    output_path: Path,
    *,
    lon_west: float = -126.85,
    lon_east: float = -121.95,
    lat_south: float = 46.0,
    lat_north: float = 49.35,
    focus_upper_depth_m: float = 280.0,
    bathy_alpha: float = 0.92,
    dpi: int = 220,
    draw_inset: bool = True,
) -> Path:
    """Write a single PNG with optional coastal regional inset."""
    cmap_mpl = _load_oslo_mpl()

    fig = plt.figure(figsize=(11.2, 9.6), facecolor="white")
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([lon_west, lon_east, lat_south, lat_north], crs=ccrs.PlateCarree())

    bath = Bathymetry()
    cf = bath.add_bathymetry_from_etopo1(
        ax=ax,
        ranges=[lon_west, lon_east, lat_south, lat_north],
        cmap=cmap_mpl,
        alpha=bathy_alpha,
        zorder=0,
        lon_plot_negative180=True,
        focus_upper_depth_m=focus_upper_depth_m,
        draw_contour_lines=False,
        extend="min",
    )

    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#d4d2ce", zorder=1)
    ax.add_feature(
        cfeature.COASTLINE.with_scale("10m"),
        linewidth=0.65,
        edgecolor="#2a2a2a",
        zorder=3,
    )
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.35, edgecolor="#666666", zorder=2)

    gl = ax.gridlines(draw_labels=True, linewidth=0.35, alpha=0.55, linestyle="--", color="#555555")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 11}
    gl.ylabel_style = {"size": 11}

    cb = fig.colorbar(cf, ax=ax, fraction=0.0315, pad=0.02, extend="min")
    cb.set_label(
        f"Depth (m) · contour levels 0–{focus_upper_depth_m:.0f} m\n"
        "(deeper offshore → under-color; ETOPO1)",
        fontsize=10,
    )

    if draw_inset:
        # Lower-right of main axes; slightly larger than before (axes fraction w × h).
        axins = ax.inset_axes(
            [0.64, 0.018, 0.33, 0.27],
            transform=ax.transAxes,
            projection=ccrs.PlateCarree(),
        )
        axins.set_extent(
            [INSET_LON_W, INSET_LON_E, INSET_LAT_S, INSET_LAT_N],
            crs=ccrs.PlateCarree(),
        )
        bath.add_bathymetry_from_etopo1(
            ax=axins,
            ranges=[INSET_LON_W, INSET_LON_E, INSET_LAT_S, INSET_LAT_N],
            cmap=cmap_mpl,
            alpha=0.95,
            zorder=0,
            lon_plot_negative180=True,
            focus_upper_depth_m=INSET_FOCUS_UPPER_DEPTH_M,
            draw_contour_lines=False,
            extend="min",
        )
        axins.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#d4d2ce", zorder=1)
        axins.add_feature(
            cfeature.COASTLINE.with_scale("10m"),
            linewidth=0.45,
            edgecolor="#1a1a1a",
            zorder=3,
        )
        axins.gridlines(draw_labels=False, linewidth=0.25, alpha=0.45, linestyle=":")
        axins.set_title("Regional context (130–120°W, 42–52°N)", fontsize=8, pad=3)
        # Red frame: main map extent on the wide-area inset.
        box_lon = [lon_west, lon_east, lon_east, lon_west, lon_west]
        box_lat = [lat_south, lat_south, lat_north, lat_north, lat_south]
        axins.plot(
            box_lon,
            box_lat,
            color="#c41e3a",
            linewidth=1.1,
            transform=ccrs.PlateCarree(),
            zorder=6,
            solid_capstyle="round",
            label="_nolegend_",
        )

    fig.suptitle(
        "Washington coast — ETOPO1 bathymetry (shallow-focused)\n"
        "Crameri Oslo (crameri:oslo) · inset = 130–120°W, 42–52°N (lower right)",
        fontsize=13,
        fontweight="semibold",
        y=0.98,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def main() -> None:
    p = argparse.ArgumentParser(description="Washington ETOPO1 bathymetry map (Crameri Oslo colormap).")
    p.add_argument(
        "--output",
        type=Path,
        default=_REPO / "StressorAnalysisV2" / "WA_state_bathymetry_etopo1_ylgnbu.png",
    )
    p.add_argument("--focus-depth-m", type=float, default=280.0, help="Upper column for contour levels (m).")
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--no-inset", action="store_true", help="Disable regional context inset.")
    p.add_argument(
        "--skip-regional-greys",
        action="store_true",
        help="Do not write the standalone regional Greys-9 map (same extent as inset).",
    )
    p.add_argument(
        "--regional-greys-output",
        type=Path,
        default=None,
        help="Path for standalone regional Greys map (default: <output stem>_regional_inset_greys.png).",
    )
    args = p.parse_args()

    out = plot_washington_bathymetry(
        args.output,
        focus_upper_depth_m=args.focus_depth_m,
        dpi=args.dpi,
        draw_inset=not args.no_inset,
    )
    print(f"Wrote {out}")

    if not args.skip_regional_greys:
        reg_out = args.regional_greys_output
        if reg_out is None:
            reg_out = args.output.parent / f"{args.output.stem}_regional_inset_greys{args.output.suffix}"
        out2 = plot_washington_bathymetry_regional_inset_standalone(
            reg_out,
            dpi=args.dpi,
        )
        print(f"Wrote {out2}")


if __name__ == "__main__":
    main()
