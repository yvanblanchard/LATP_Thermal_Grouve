"""
Comparison of the former optical model (real refractive index, lossless dielectric
Fresnel) with the new complex-index model (N = n + ik) in ray_tracing_2d.py.

Run:  python test_optical_model_comparison.py            (checks + summary + figure)
      python -m pytest test_optical_model_comparison.py   (checks only)

Parts
-----
1. Unit checks of the Fresnel function
   - k = 0 reproduces the former model exactly
   - normal incidence: R = ((n-1)^2 + k^2) / ((n+1)^2 + k^2)
   - 0 <= R <= 1 for all angles, R -> 1 at grazing incidence
2. Ray-tracing comparison: absorbed flux on substrate and tape (roller),
   former model vs. new model for several k values.
"""
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ray_tracing_2d import (
    VectorizedLaser, VectorizedRoller, VectorizedSubstrate, VectorizedRayTracer
)

# Same geometry / process parameters as thermal_model_Grouve.py
ROLLER_RADIUS = 40e-3
SUBSTRATE_LENGTH = 100e-3
LASER = dict(source_length=30e-3, source_center=np.array([-150e-3, 60.50e-3]),
             source_angle=22.0, num_rays=5000, total_power=1.0)
N_TAPE = 1.5
MAX_REFLECTIONS = 3
THRESHOLD = 1e-6
NUM_POINTS = 200

# (label, k) -> new model cases; k = 0 must match the former model
K_CASES = [0.0, 0.1, 0.5, 1.0]


# Kapton film substrate, optical constants read from refractiveindex.info
# (Arakawa et al. 1981: n 0.02-1.49 um, k 0.02-0.43 um). k is not tabulated above
# 0.43 um, so it is taken as 0 in the visible/NIR.
KAPTON = {
    "0.5876um": dict(n=1.7496, k=0.0),        # n read at 0.5876 um
    "0.28um (UV)": dict(n=1.99375, k=0.403634),  # n, k read at 0.28 um (illustrative, k != 0)
}


def _tracer(n, k, legacy, substrate_nk=None):
    """n, k: roller (tape). substrate_nk: optional (n, k) for the substrate film."""
    laser = VectorizedLaser(**LASER)
    roller = VectorizedRoller(radius=ROLLER_RADIUS, refractive_index=n, extinction_coeff=k)
    ns, ks = substrate_nk if substrate_nk is not None else (n, k)
    substrate = VectorizedSubstrate(length=SUBSTRATE_LENGTH, refractive_index=ns,
                                    extinction_coeff=ks)
    return VectorizedRayTracer(laser, roller, substrate, max_reflections=MAX_REFLECTIONS,
                               min_power_threshold_fraction=THRESHOLD,
                               legacy_fresnel=legacy)


def compute_fluxes(n=N_TAPE, k=0.0, legacy=False, substrate_nk=None):
    """Return dict with distance/flux arrays for substrate and tape (roller)."""
    tracer = _tracer(n, k, legacy, substrate_nk)
    tracer.trace_all_rays_vectorized()
    s_dist, _, _, _, s_flux = tracer.calculate_irradiance_by_generation_vectorized(
        tracer.substrate, NUM_POINTS)
    t_dist, _, _, _, t_flux = tracer.calculate_irradiance_by_generation_vectorized(
        tracer.roller, NUM_POINTS)
    return dict(s_dist=s_dist, s_flux=np.maximum(s_flux, 0.0),
                t_dist=t_dist, t_flux=np.maximum(t_flux, 0.0))


# ---------------------------------------------------------------------------
# 1. Unit checks
# ---------------------------------------------------------------------------
def _fresnel(n, k, theta, legacy=False):
    return _tracer(n, k, legacy).fresnel_reflectance_vectorized(theta, 1.0, complex(n, k))


def test_k_zero_matches_former_model():
    theta = np.radians(np.linspace(0, 89.5, 200))
    for n in (1.2, 1.5, 1.8, 2.4):
        new = _fresnel(n, 0.0, theta)
        old = _fresnel(n, 0.0, theta, legacy=True)
        assert np.allclose(new, old, atol=1e-12), f"n={n}: max diff {np.max(abs(new-old))}"


def test_normal_incidence_closed_form():
    for n, k in [(1.5, 0.0), (1.5, 0.5), (2.2, 1.2), (3.0, 0.01)]:
        R = _fresnel(n, k, np.array([0.0]))[0]
        expected = ((n - 1) ** 2 + k ** 2) / ((n + 1) ** 2 + k ** 2)
        assert abs(R - expected) < 1e-12, (n, k, R, expected)


def test_reflectance_bounds_and_grazing():
    theta = np.radians(np.linspace(0, 90, 361))
    for n, k in [(1.5, 0.0), (1.5, 1.0), (2.2, 1.2)]:
        R = _fresnel(n, k, theta)
        assert np.all(R >= -1e-12) and np.all(R <= 1 + 1e-12)
        assert R[-1] > 0.999  # grazing incidence -> total reflection


def test_k_increases_normal_reflectance():
    R0 = _fresnel(1.5, 0.0, np.array([0.0]))[0]
    R1 = _fresnel(1.5, 1.0, np.array([0.0]))[0]
    assert R1 > R0


def test_ray_tracing_k_zero_equals_legacy():
    old = compute_fluxes(legacy=True)
    new = compute_fluxes(k=0.0)
    assert np.allclose(old["s_flux"], new["s_flux"], atol=1e-10)
    assert np.allclose(old["t_flux"], new["t_flux"], atol=1e-10)


def test_kapton_substrate():
    """Kapton film substrate (optical constants from the image), tape n=N_TAPE."""
    # visible case from the image: k = 0 -> new model must equal the former one
    kv = KAPTON["0.5876um"]
    old = compute_fluxes(legacy=True, substrate_nk=(kv["n"], kv["k"]))
    new = compute_fluxes(substrate_nk=(kv["n"], kv["k"]))
    assert np.allclose(old["s_flux"], new["s_flux"], atol=1e-10)
    assert np.allclose(old["t_flux"], new["t_flux"], atol=1e-10)
    # normal-incidence reflectance of the Kapton surface
    R = _fresnel(kv["n"], 0.0, np.array([0.0]))[0]
    assert abs(R - ((kv["n"] - 1) / (kv["n"] + 1)) ** 2) < 1e-12

    # UV case (k = 0.4036): absorption by Fresnel must drop and R must rise vs k = 0
    ku = KAPTON["0.28um (UV)"]
    R_uv = _fresnel(ku["n"], ku["k"], np.array([0.0]))[0]
    R_uv_k0 = _fresnel(ku["n"], 0.0, np.array([0.0]))[0]
    assert R_uv > R_uv_k0
    uv = compute_fluxes(substrate_nk=(ku["n"], ku["k"]))
    uv_k0 = compute_fluxes(substrate_nk=(ku["n"], 0.0))
    assert uv["s_flux"].max() < uv_k0["s_flux"].max()


# ---------------------------------------------------------------------------
# 2. Comparison report + figure
# ---------------------------------------------------------------------------
def _summary(name, dist, flux):
    mask = flux > 0.01 * flux.max() if flux.max() > 0 else np.zeros_like(flux, bool)
    extent = (dist[mask].min() * 1e3, dist[mask].max() * 1e3) if mask.any() else (0, 0)
    # integral over arc length (distances are monotonic but may be descending)
    order = np.argsort(dist)
    integral = np.trapezoid(flux[order], dist[order])
    return f"{name:<22s} peak={flux.max():.4f}  integral={integral*1e3:.4f} (W.mm)  " \
           f"extent=[{extent[0]:.1f}, {extent[1]:.1f}] mm"


def run_comparison(out_png="optical_model_comparison.png"):
    former = compute_fluxes(legacy=True)
    results = {k: compute_fluxes(k=k) for k in K_CASES}

    print("\n" + "=" * 90)
    print(f"Optical model comparison  (n={N_TAPE}, alpha={LASER['source_angle']} deg, "
          f"{LASER['num_rays']} rays)")
    print("=" * 90)
    for part, d, f in (("SUBSTRATE", "s_dist", "s_flux"), ("TAPE", "t_dist", "t_flux")):
        print(f"\n--- {part} ---")
        print(_summary("former (real n)", former[d], former[f]))
        for k in K_CASES:
            r = results[k]
            line = _summary(f"new  n+ik, k={k}", r[d], r[f])
            rel = np.max(np.abs(r[f] - former[f])) / max(former[f].max(), 1e-30)
            print(f"{line}  max|diff|/peak={rel:.3%}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for col, (title, d, f) in enumerate((("Substrate", "s_dist", "s_flux"),
                                         ("Tape (roller)", "t_dist", "t_flux"))):
        ax = axes[0, col]
        ax.plot(former[d] * 1e3, former[f], "k--", lw=2, label="former (real n)")
        for k in K_CASES:
            ax.plot(results[k][d] * 1e3, results[k][f], lw=1.2, label=f"new, k={k}")
        ax.set_title(f"{title}: absorbed flux (n={N_TAPE})")
        ax.set_xlabel("distance from nip-point [mm]")
        ax.set_ylabel("absorbed flux [W]")
        ax.grid(alpha=0.3)
        ax.legend()

        ax = axes[1, col]
        for k in K_CASES:
            ax.plot(results[k][d] * 1e3, results[k][f] - former[f], lw=1.2,
                    label=f"k={k}")
        ax.set_title(f"{title}: new - former")
        ax.set_xlabel("distance from nip-point [mm]")
        ax.set_ylabel("difference [W]")
        ax.grid(alpha=0.3)
        ax.legend()

    run_kapton_comparison(former, out_png.replace(".png", "_kapton.png"))
    fig.savefig(out_png, dpi=130)
    print(f"\nFigure saved to {os.path.abspath(out_png)}")


def run_kapton_comparison(_unused=None, out_png="optical_model_comparison_kapton.png"):
    """Kapton film substrate: former model (real n only) vs new model (n + ik)."""
    print("\n" + "=" * 90)
    print(f"Kapton substrate (tape/roller n={N_TAPE}); constants from refractiveindex.info image")
    print("=" * 90)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for label, c in KAPTON.items():
        nk = (c["n"], c["k"])
        former = compute_fluxes(legacy=True, substrate_nk=nk)
        new = compute_fluxes(substrate_nk=nk)
        print(f"\n--- Kapton @ {label}: n={c['n']}, k={c['k']} ---")
        for part, d, f in (("substrate", "s_dist", "s_flux"), ("tape", "t_dist", "t_flux")):
            print(_summary(f"former {part}", former[d], former[f]))
            rel = np.max(np.abs(new[f] - former[f])) / max(former[f].max(), 1e-30)
            print(_summary(f"new    {part}", new[d], new[f]) + f"  max|diff|/peak={rel:.3%}")
        for ax, (d, f, t) in zip(axes, (("s_dist", "s_flux", "Kapton substrate"),
                                        ("t_dist", "t_flux", "Tape (roller)"))):
            line, = ax.plot(new[d] * 1e3, new[f], label=f"new, {label}")
            ax.plot(former[d] * 1e3, former[f], "--", color=line.get_color(),
                    label=f"former, {label}")
            ax.set_title(f"{t}: absorbed flux")
            ax.set_xlabel("distance from nip-point [mm]")
            ax.set_ylabel("absorbed flux [W]")
            ax.grid(alpha=0.3)
            ax.legend()
    fig.savefig(out_png, dpi=130)
    print(f"\nFigure saved to {os.path.abspath(out_png)}")


if __name__ == "__main__":
    for fn in (test_k_zero_matches_former_model, test_normal_incidence_closed_form,
               test_reflectance_bounds_and_grazing, test_k_increases_normal_reflectance,
               test_ray_tracing_k_zero_equals_legacy, test_kapton_substrate):
        fn()
        print(f"PASS  {fn.__name__}")
    run_comparison()
