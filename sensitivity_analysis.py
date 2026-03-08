"""
sensitivity_analysis.py
=======================
Sensitivity study for the 2-D ray-tracing model.

Two parameters are swept independently around a nominal operating point:
  - Laser incidence angle  (degrees, angle w.r.t. vertical Z axis)
  - Source height distance (mm, Z-coordinate of the laser source centre)

For each sweep the script produces:
  Set 1 (angle sweep)  : substrate flux profiles | tape flux profiles
  Set 2 (height sweep) : substrate flux profiles | tape flux profiles
  Set 3 (shared)       : percentage-sensitivity plot
                         (% input variation → % output variation)
  Set 4 (both sweeps)  : substrate-vs-tape flux gradient profiles
                         (difference and ratio at same arc-length from nip)
  Set 5 (shared)       : percentage-sensitivity of gradient scalar metrics
"""

import contextlib
import io

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap

from ray_tracing_2d import (
    VectorizedLaser,
    VectorizedRayTracer,
    VectorizedRoller,
    VectorizedSubstrate,
)

# ---------------------------------------------------------------------------
# Nominal / fixed simulation parameters
# ---------------------------------------------------------------------------
NOMINAL_LASER_ANGLE = 22.0          # degrees
NOMINAL_SOURCE_Z    = 55e-3         # m  (height above substrate plane z=0)
NOMINAL_SOURCE_Y    = -150e-3       # m  (fixed lateral position)
SOURCE_LENGTH       = 30e-3         # m
ROLLER_RADIUS       = 40e-3         # m
SUBSTRATE_LENGTH    = 100e-3        # m
REFRACTIVE_INDEX    = 1.5
NUM_RAYS            = 5_000         # trade-off speed / results
MAX_REFLECTIONS     = 3
IRRADIANCE_PTS      = 500           # discretisation points per surface

# ---------------------------------------------------------------------------
# Sensitivity sweep ranges  (include the nominal value)
# ---------------------------------------------------------------------------
ANGLE_RANGE  = np.arange(19.0, 29.0, 1.0)                   # degrees  19→28
HEIGHT_RANGE = np.arange(45.0, 75.0, 5.0) * 1e-3            # metres   45→70 mm

# Deduplicate and sort
ANGLE_RANGE  = np.unique(ANGLE_RANGE)
HEIGHT_RANGE = np.unique(HEIGHT_RANGE)

# ---------------------------------------------------------------------------
# Core simulation helper
# ---------------------------------------------------------------------------

def _run_simulation(laser_angle: float,
                    source_z: float,
                    source_y: float = NOMINAL_SOURCE_Y) -> dict:
    """
    Execute one ray-tracing run and return flux arrays + scalar metrics.

    Returns
    -------
    dict with keys:
        substrate_dist, substrate_flux, substrate_max_extent,
        roller_dist,    roller_flux,    roller_max_extent,
        max_sub_flux,   max_roll_flux
    """
    laser = VectorizedLaser(
        source_length=SOURCE_LENGTH,
        source_center=np.array([source_y, source_z]),
        source_angle=laser_angle,
        num_rays=NUM_RAYS,
        total_power=1.0,
    )
    roller    = VectorizedRoller(radius=ROLLER_RADIUS, refractive_index=REFRACTIVE_INDEX)
    substrate = VectorizedSubstrate(length=SUBSTRATE_LENGTH, refractive_index=REFRACTIVE_INDEX)
    tracer    = VectorizedRayTracer(
        laser, roller, substrate,
        max_reflections=MAX_REFLECTIONS,
        min_power_threshold_fraction=1e-10,
    )

    with contextlib.redirect_stdout(io.StringIO()):
        tracer.trace_all_rays_vectorized()
        sub_dist, _, _, sub_max_ext, sub_flux = (
            tracer.calculate_irradiance_by_generation_vectorized(substrate, IRRADIANCE_PTS))
        roll_dist, _, _, roll_max_ext, roll_flux = (
            tracer.calculate_irradiance_by_generation_vectorized(roller, IRRADIANCE_PTS))

    return {
        "substrate_dist":       sub_dist,
        "substrate_flux":       sub_flux,
        "substrate_max_extent": sub_max_ext,
        "roller_dist":          roll_dist,
        "roller_flux":          roll_flux,
        "roller_max_extent":    roll_max_ext,
        "max_sub_flux":         float(np.max(sub_flux)),
        "max_roll_flux":        float(np.max(roll_flux)),
    }


# ---------------------------------------------------------------------------
# Flux-gradient helpers
# ---------------------------------------------------------------------------

# Small epsilon used to guard divisions by zero in ratio computation.
_RATIO_EPS = 1e-12

# Number of points on the common arc-length grid used for gradient analysis.
_GRADIENT_PTS = 300


def _enrich_with_gradient(res: dict) -> None:
    """
    Enrich a result dict (in-place) with substrate-vs-tape flux gradient data.

    Both flux profiles are interpolated onto a shared arc-length grid that
    spans the *overlap* of the illuminated regions of both surfaces.  The
    following keys are added:

    gradient_dist   : (N,) common arc-length array from nip point (m)
    gradient_diff   : (N,) substrate_flux - tape_flux  (W)
    gradient_ratio  : (N,) substrate_flux / tape_flux  (dimensionless, NaN
                      where tape flux is below threshold)
    peak_abs_diff   : scalar, max |gradient_diff|  (W)
    mean_abs_diff   : scalar, mean |gradient_diff| over illuminated zone (W)
    peak_ratio      : scalar, max ratio where both surfaces are illuminated
    """
    sub_dist  = res["substrate_dist"]
    sub_flux  = res["substrate_flux"]
    roll_dist = res["roller_dist"]
    roll_flux = res["roller_flux"]

    # Both distance arrays may be descending (distance from nip computed via
    # abs(y) or arc-length inverted).  np.interp requires ascending xp.
    sub_order  = np.argsort(sub_dist)
    sub_dist   = sub_dist[sub_order]
    sub_flux   = sub_flux[sub_order]

    roll_order = np.argsort(roll_dist)
    roll_dist  = roll_dist[roll_order]
    roll_flux  = roll_flux[roll_order]

    # Illuminated extent: region where flux exceeds 1 % of each surface peak.
    sub_threshold  = np.max(sub_flux)  * 0.01
    roll_threshold = np.max(roll_flux) * 0.01

    sub_illum  = sub_dist[sub_flux  > sub_threshold]
    roll_illum = roll_dist[roll_flux > roll_threshold]

    if sub_illum.size == 0 or roll_illum.size == 0:
        # No meaningful overlap — fill with zeros
        common = np.linspace(0, 1e-3, _GRADIENT_PTS)
        res["gradient_dist"]  = common
        res["gradient_diff"]  = np.zeros(_GRADIENT_PTS)
        res["gradient_ratio"] = np.full(_GRADIENT_PTS, np.nan)
        res["peak_abs_diff"]  = 0.0
        res["mean_abs_diff"]  = 0.0
        res["peak_ratio"]     = np.nan
        return

    # Common range: overlap of both illuminated extents starting from nip.
    x_start = max(np.min(sub_illum),  np.min(roll_illum))
    x_end   = min(np.max(sub_illum),  np.max(roll_illum))

    if x_end <= x_start:
        x_start = 0.0
        x_end   = min(np.max(sub_dist), np.max(roll_dist))

    common = np.linspace(x_start, x_end, _GRADIENT_PTS)

    # Interpolate both profiles onto common grid (zero outside original range).
    sub_interp  = np.interp(common, sub_dist,  sub_flux,  left=0.0, right=0.0)
    roll_interp = np.interp(common, roll_dist, roll_flux, left=0.0, right=0.0)

    diff  = sub_interp - roll_interp
    denom = np.where(roll_interp > _RATIO_EPS, roll_interp, np.nan)
    ratio = sub_interp / denom

    # Scalar metrics — only over jointly illuminated zone.
    both_lit = (sub_interp > sub_threshold) & (roll_interp > roll_threshold)
    if both_lit.any():
        peak_abs_diff = float(np.max(np.abs(diff[both_lit])))
        mean_abs_diff = float(np.mean(np.abs(diff[both_lit])))
        peak_ratio    = float(np.nanmax(ratio[both_lit]))
    else:
        peak_abs_diff = float(np.max(np.abs(diff)))
        mean_abs_diff = float(np.mean(np.abs(diff)))
        peak_ratio    = float(np.nanmax(ratio))

    res["gradient_dist"]  = common
    res["gradient_diff"]  = diff
    res["gradient_ratio"] = ratio
    res["peak_abs_diff"]  = peak_abs_diff
    res["mean_abs_diff"]  = mean_abs_diff
    res["peak_ratio"]     = peak_ratio


# ---------------------------------------------------------------------------
# Sweep drivers
# ---------------------------------------------------------------------------

def sweep_angle() -> list[dict]:
    """Sweep laser incidence angle; return list of result dicts."""
    results = []
    for angle in ANGLE_RANGE:
        print(f"  [angle sweep]  angle = {angle:5.1f}°  …", end="", flush=True)
        res = _run_simulation(laser_angle=angle, source_z=NOMINAL_SOURCE_Z)
        res["angle"] = angle
        res["label"] = f"θ = {angle:.1f}°"
        _enrich_with_gradient(res)
        results.append(res)
        print(f"  max sub flux = {res['max_sub_flux']:.4f}  |  "
              f"max roll flux = {res['max_roll_flux']:.4f}  |  "
              f"peak |Δflux| = {res['peak_abs_diff']:.4f}")
    return results


def sweep_height() -> list[dict]:
    """Sweep source height (source_z); return list of result dicts."""
    results = []
    for height in HEIGHT_RANGE:
        print(f"  [height sweep]  h = {height * 1e3:5.1f} mm  …", end="", flush=True)
        res = _run_simulation(laser_angle=NOMINAL_LASER_ANGLE, source_z=height)
        res["height"] = height
        res["label"] = f"h = {height * 1e3:.0f} mm"
        _enrich_with_gradient(res)
        results.append(res)
        print(f"  max sub flux = {res['max_sub_flux']:.4f}  |  "
              f"max roll flux = {res['max_roll_flux']:.4f}  |  "
              f"peak |Δflux| = {res['peak_abs_diff']:.4f}")
    return results


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _discrete_colors(n: int, cmap_name: str = "plasma") -> list:
    cmap = get_cmap(cmap_name, n)
    return [cmap(i) for i in range(n)]


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _flux_profile_axes(ax, results: list[dict],
                       surface_key: str,
                       dist_key: str,
                       extent_key: str,
                       title: str,
                       xlabel: str) -> None:
    """Draw one set of flux profile curves on *ax*."""
    colors = _discrete_colors(len(results))
    global_max_ext = max(r[extent_key] for r in results) * 1e3  # mm

    for res, color in zip(results, colors):
        dist = res[dist_key] * 1e3        # → mm
        flux = res[surface_key]
        ax.plot(dist, flux, color=color, linewidth=1.4, label=res["label"])

    ax.set_xlim(left=0, right=global_max_ext * 1.05)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Absorbed power flux (W)", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.7)
    ax.grid(True, alpha=0.25)


def plot_angle_sensitivity_profiles(angle_results: list[dict],
                                    save_path: str | None = None) -> None:
    """
    Figure 1 — two panels side by side showing flux profiles for each
    laser angle (substrate left, tape/roller right).
    """
    fig, (ax_sub, ax_roll) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Sensitivity to Laser Incidence Angle\n"
        f"(source height fixed at {NOMINAL_SOURCE_Z * 1e3:.0f} mm, "
        f"one curve per angle)",
        fontsize=12,
    )

    _flux_profile_axes(
        ax_sub, angle_results,
        surface_key="substrate_flux", dist_key="substrate_dist",
        extent_key="substrate_max_extent",
        title="Substrate — Flux Profile",
        xlabel="Distance from nip point (mm)",
    )
    _flux_profile_axes(
        ax_roll, angle_results,
        surface_key="roller_flux", dist_key="roller_dist",
        extent_key="roller_max_extent",
        title="Tape (roller) — Flux Profile",
        xlabel="Arc distance from nip point (mm)",
    )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.show()


def plot_height_sensitivity_profiles(height_results: list[dict],
                                     save_path: str | None = None) -> None:
    """
    Figure 2 — two panels side by side showing flux profiles for each
    source height (substrate left, tape/roller right).
    """
    fig, (ax_sub, ax_roll) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Sensitivity to Source Height Distance\n"
        f"(laser angle fixed at {NOMINAL_LASER_ANGLE:.1f}°, "
        f"one curve per height)",
        fontsize=12,
    )

    _flux_profile_axes(
        ax_sub, height_results,
        surface_key="substrate_flux", dist_key="substrate_dist",
        extent_key="substrate_max_extent",
        title="Substrate — Flux Profile",
        xlabel="Distance from nip point (mm)",
    )
    _flux_profile_axes(
        ax_roll, height_results,
        surface_key="roller_flux", dist_key="roller_dist",
        extent_key="roller_max_extent",
        title="Tape (roller) — Flux Profile",
        xlabel="Arc distance from nip point (mm)",
    )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Percentage-sensitivity helpers
# ---------------------------------------------------------------------------

def _pct_variation(values: np.ndarray, nominal: float) -> np.ndarray:
    """Return (value - nominal) / nominal * 100  for each element."""
    return (values - nominal) / nominal * 100.0


def _sensitivity_index(input_pct: np.ndarray, output_pct: np.ndarray) -> float:
    """
    Estimate local sensitivity index S = Δ(output%) / Δ(input%) around 0.

    Uses a linear regression on the central ±20% of the input range.
    """
    mask = np.abs(input_pct) <= 20.0
    if mask.sum() < 2:
        mask = np.ones(len(input_pct), dtype=bool)
    coeffs = np.polyfit(input_pct[mask], output_pct[mask], 1)
    return float(coeffs[0])


def _draw_pct_panel(ax,
                    input_pct: np.ndarray,
                    sub_pct: np.ndarray,
                    roll_pct: np.ndarray,
                    param_label: str,
                    nominal_label: str,
                    peak_diff_pct: np.ndarray | None = None,
                    mean_diff_pct: np.ndarray | None = None) -> None:
    """Draw one percentage-sensitivity panel, optionally with gradient metrics."""
    ax.plot(input_pct, sub_pct,  "o-",  color="steelblue",  linewidth=1.8,
            label="Substrate max flux")
    ax.plot(input_pct, roll_pct, "s--", color="darkorange", linewidth=1.8,
            label="Tape max flux")

    if peak_diff_pct is not None:
        ax.plot(input_pct, peak_diff_pct, "^-", color="mediumseagreen",
                linewidth=1.8, label="Peak |Δflux| (sub−tape)")
    if mean_diff_pct is not None:
        ax.plot(input_pct, mean_diff_pct, "v:", color="mediumpurple",
                linewidth=1.8, label="Mean |Δflux| (sub−tape)")

    # Annotate sensitivity indices
    si_sub  = _sensitivity_index(input_pct, sub_pct)
    si_roll = _sensitivity_index(input_pct, roll_pct)
    annotation = (f"S_substrate ≈ {si_sub:+.2f} %/%\n"
                  f"S_tape      ≈ {si_roll:+.2f} %/%")
    if peak_diff_pct is not None:
        si_peak = _sensitivity_index(input_pct, peak_diff_pct)
        annotation += f"\nS_peak|Δf|  ≈ {si_peak:+.2f} %/%"
    if mean_diff_pct is not None:
        si_mean = _sensitivity_index(input_pct, mean_diff_pct)
        annotation += f"\nS_mean|Δf|  ≈ {si_mean:+.2f} %/%"

    ax.text(0.04, 0.96, annotation,
            transform=ax.transAxes, fontsize=7.5,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="goldenrod", alpha=0.9))

    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    ax.axvline(0, color="grey", linewidth=0.8, linestyle=":")
    ax.set_xlabel(f"{param_label} variation from nominal (%)", fontsize=10)
    ax.set_ylabel("Output metric variation from nominal (%)", fontsize=10)
    ax.set_title(f"Sensitivity to {param_label}\n(nominal: {nominal_label})",
                 fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)


def _gradient_profile_axes(ax, results: list[dict], title: str,
                            show_ratio: bool = False) -> None:
    """
    Draw substrate-vs-tape flux gradient profiles on *ax*.

    By default draws the signed difference (substrate − tape).  If
    *show_ratio* is True, draws the ratio (substrate / tape) instead.
    """
    colors = _discrete_colors(len(results))
    for res, color in zip(results, colors):
        dist = res["gradient_dist"] * 1e3   # → mm
        if show_ratio:
            ydata = res["gradient_ratio"]
            ylabel = "Flux ratio  substrate / tape  (—)"
        else:
            ydata = res["gradient_diff"]
            ylabel = "Flux difference  substrate − tape  (W)"
        ax.plot(dist, ydata, color=color, linewidth=1.4, label=res["label"])

    ax.axhline(0 if not show_ratio else 1,
               color="grey", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Arc-length from nip point (mm)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.7)
    ax.grid(True, alpha=0.25)


def plot_flux_gradient_profiles(angle_results: list[dict],
                                height_results: list[dict],
                                save_path: str | None = None) -> None:
    """
    Figure 4 — 2 × 2 grid showing the substrate-vs-tape flux gradient.

    Row 1 : angle sweep   — difference (left) | ratio (right)
    Row 2 : height sweep  — difference (left) | ratio (right)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Substrate vs. Tape Flux Gradient at Same Arc-Length from Nip Point",
        fontsize=13,
    )

    _gradient_profile_axes(
        axes[0, 0], angle_results,
        title=f"Angle sweep — Difference\n"
              f"(height fixed at {NOMINAL_SOURCE_Z * 1e3:.0f} mm)",
        show_ratio=False,
    )
    _gradient_profile_axes(
        axes[0, 1], angle_results,
        title=f"Angle sweep — Ratio\n"
              f"(height fixed at {NOMINAL_SOURCE_Z * 1e3:.0f} mm)",
        show_ratio=True,
    )
    _gradient_profile_axes(
        axes[1, 0], height_results,
        title=f"Height sweep — Difference\n"
              f"(angle fixed at {NOMINAL_LASER_ANGLE:.1f}°)",
        show_ratio=False,
    )
    _gradient_profile_axes(
        axes[1, 1], height_results,
        title=f"Height sweep — Ratio\n"
              f"(angle fixed at {NOMINAL_LASER_ANGLE:.1f}°)",
        show_ratio=True,
    )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.show()


def plot_percentage_sensitivity(angle_results: list[dict],
                                height_results: list[dict],
                                save_path: str | None = None) -> None:
    """
    Figure 3 — percentage sensitivity plots.

    Left panel  : laser angle sensitivity
    Right panel : source height sensitivity

    Each panel shows curves for substrate max flux and tape max flux.
    The sensitivity index S (% output per % input) is printed on the plot.
    """
    # ── angle ──────────────────────────────────────────────────────────────
    angles      = np.array([r["angle"] for r in angle_results])
    sub_angle   = np.array([r["max_sub_flux"]  for r in angle_results])
    roll_angle  = np.array([r["max_roll_flux"] for r in angle_results])
    peak_diff_a = np.array([r["peak_abs_diff"] for r in angle_results])
    mean_diff_a = np.array([r["mean_abs_diff"] for r in angle_results])

    ref_a     = NOMINAL_LASER_ANGLE
    ref_idx_a = int(np.argmin(np.abs(angles - ref_a)))
    angle_pct       = _pct_variation(angles,      ref_a)
    sub_angle_pct   = _pct_variation(sub_angle,   sub_angle[ref_idx_a])
    roll_angle_pct  = _pct_variation(roll_angle,  roll_angle[ref_idx_a])
    peak_diff_a_pct = _pct_variation(peak_diff_a, peak_diff_a[ref_idx_a])
    mean_diff_a_pct = _pct_variation(mean_diff_a, mean_diff_a[ref_idx_a])

    # ── height ─────────────────────────────────────────────────────────────
    heights     = np.array([r["height"] for r in height_results])
    sub_height  = np.array([r["max_sub_flux"]  for r in height_results])
    roll_height = np.array([r["max_roll_flux"] for r in height_results])
    peak_diff_h = np.array([r["peak_abs_diff"] for r in height_results])
    mean_diff_h = np.array([r["mean_abs_diff"] for r in height_results])

    ref_h     = NOMINAL_SOURCE_Z
    ref_idx_h = int(np.argmin(np.abs(heights - ref_h)))
    height_pct      = _pct_variation(heights,     ref_h)
    sub_height_pct  = _pct_variation(sub_height,  sub_height[ref_idx_h])
    roll_height_pct = _pct_variation(roll_height, roll_height[ref_idx_h])
    peak_diff_h_pct = _pct_variation(peak_diff_h, peak_diff_h[ref_idx_h])
    mean_diff_h_pct = _pct_variation(mean_diff_h, mean_diff_h[ref_idx_h])

    # ── build figure ───────────────────────────────────────────────────────
    fig, (ax_a, ax_h) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Percentage Sensitivity Analysis\n"
        "% variation of output metric vs. % variation of input parameter",
        fontsize=12,
    )

    _draw_pct_panel(ax_a,
                    angle_pct, sub_angle_pct, roll_angle_pct,
                    param_label="Laser angle",
                    nominal_label=f"{NOMINAL_LASER_ANGLE:.1f}°",
                    peak_diff_pct=peak_diff_a_pct,
                    mean_diff_pct=mean_diff_a_pct)

    _draw_pct_panel(ax_h,
                    height_pct, sub_height_pct, roll_height_pct,
                    param_label="Source height",
                    nominal_label=f"{NOMINAL_SOURCE_Z * 1e3:.0f} mm",
                    peak_diff_pct=peak_diff_h_pct,
                    mean_diff_pct=mean_diff_h_pct)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 62)
    print("  Ray-Tracing 2D — Sensitivity Analysis")
    print("=" * 62)
    print(f"  Nominal angle  : {NOMINAL_LASER_ANGLE} °")
    print(f"  Nominal height : {NOMINAL_SOURCE_Z * 1e3:.0f} mm")
    print(f"  Num rays       : {NUM_RAYS}")
    print(f"  Max reflections: {MAX_REFLECTIONS}")
    print()

    # ── Sweep 1: laser angle ───────────────────────────────────────────────
    print("Running laser angle sweep …")
    angle_results = sweep_angle()
    print()

    # ── Sweep 2: source height ─────────────────────────────────────────────
    print("Running source height sweep …")
    height_results = sweep_height()
    print()

    # ── Plots ──────────────────────────────────────────────────────────────
    print("Generating plots …")

    plot_angle_sensitivity_profiles(
        angle_results,
        save_path="sensitivity_angle_flux_profiles.png",
    )
    plot_height_sensitivity_profiles(
        height_results,
        save_path="sensitivity_height_flux_profiles.png",
    )
    plot_percentage_sensitivity(
        angle_results,
        height_results,
        save_path="sensitivity_percentage.png",
    )
    plot_flux_gradient_profiles(
        angle_results,
        height_results,
        save_path="sensitivity_flux_gradient_profiles.png",
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
