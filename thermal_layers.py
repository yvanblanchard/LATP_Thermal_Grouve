"""
Layered 1D Thermal Primitives for the LATP Process Model
=========================================================

Shared building blocks used by the consolidation-phase and cooling-phase
models. A LATP thermal history involves three successive configurations of the
*same* material stack:

    Phase 1  HEATING        substrate alone (laser flux on top surface)
                            tape alone, on the roller (laser flux on its
                            bonding face)
    Phase 2  CONSOLIDATION  roller / tape / substrate, joined by contact
                            resistances
    Phase 3  COOLING        tape / substrate, roller lifted off

All three are the same mathematical object: a chain of homogeneous layers
joined by zero-thickness thermal contact resistances, with a convective +
imposed-flux boundary condition at each outer face. `LayerStack` implements
that object once; the phase modules only supply geometry and boundary data.


DISCRETISATION
==============

Finite-volume (energy balance) in space, backward Euler in time. Node `i`
owns a control volume of thickness `dz_cv[i]`: `dz` for a node interior to a
layer, `dz/2` for a node on a layer face. Summed over a layer this recovers
exactly the layer thickness, so the scheme is conservative.

Neighbouring nodes exchange heat through a conductance `G[i]` linking node `i`
to node `i+1`  [W/(m^2.K)]:

    inside a layer      G = k / dz
    across an interface G = h_contact = 1 / R_tc

The implicit system for every node is then the same three-term expression:

    -G[i-1] T'[i-1]
    + ( C[i]/dt + G[i-1] + G[i] + h_bc[i] ) T'[i]
    - G[i]   T'[i+1]
        = C[i]/dt T[i] + h_bc[i] T_ref[i] + Q[i]

with C[i] = rho*cp*dz_cv[i]  [J/(m^2.K)], and the `h_bc` / `T_ref` / `Q`
terms present only on the two outer faces (i = 0 and i = N-1). G[-1] and
G[N-1] do not exist and are taken as zero.

This is the same surface energy balance used by `thermal_model_Grouve.py`
(`ThermalSolver._apply_boundary_conditions`), generalised to several layers.

The matrix is tridiagonal, symmetric and positive definite, and -- because
every coefficient depends only on `dt` and material data, never on `T` -- it is
constant in time. It is therefore assembled once per distinct `dt` and only
the right-hand side is rebuilt each step, which is what makes a multi-position
sweep affordable.

SIGN CONVENTION
===============
`Q` is heat flowing *into* the domain through the face  [W/m^2].
"""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

try:
    from scipy.linalg import solve_banded
    HAS_SCIPY = True
except ImportError:
    print("Warning: scipy not available, falling back to dense linear solves")
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Material presets
# ---------------------------------------------------------------------------

# Through-thickness properties. The APC-2 entry is deliberately identical to
# MATERIAL_PROPS in thermal_model_Grouve.py so the phases share one material.
MATERIAL_PRESETS = {
    'apc2_cf_peek':    dict(rho=1600.0, cp=1000.0, k=0.7),
    'silicone_roller': dict(rho=1200.0, cp=1500.0, k=0.25),
    'steel_roller':    dict(rho=7850.0, cp=490.0,  k=45.0),
}


def material(name: str) -> dict:
    """Look up a preset from MATERIAL_PRESETS."""
    try:
        return dict(MATERIAL_PRESETS[name])
    except KeyError:
        raise KeyError(
            f"unknown material {name!r}; available: {sorted(MATERIAL_PRESETS)}"
        ) from None


# ---------------------------------------------------------------------------
# Layer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Layer:
    """
    One homogeneous layer, discretised on `N` uniformly spaced nodes.

    Node 0 sits on the layer's top face, node N-1 on its bottom face, so a
    stack shares no nodes between layers: the two nodes either side of an
    interface are distinct and coupled through a contact resistance.
    """
    name:      str
    rho:       float   # kg/m^3
    cp:        float   # J/(kg.K)
    k:         float   # W/(m.K)  through-thickness
    thickness: float   # m
    N:         int = 10

    def __post_init__(self):
        if self.N < 2:
            raise ValueError(f"layer {self.name!r}: N must be >= 2, got {self.N}")
        if self.thickness <= 0:
            raise ValueError(
                f"layer {self.name!r}: thickness must be > 0, got {self.thickness}"
            )

    @property
    def dz(self) -> float:
        return self.thickness / (self.N - 1)

    @property
    def alpha(self) -> float:
        """Thermal diffusivity [m^2/s]."""
        return self.k / (self.rho * self.cp)

    @property
    def rho_cp(self) -> float:
        """Volumetric heat capacity [J/(m^3.K)]."""
        return self.rho * self.cp

    def diffusion_dt(self, safety: float = 0.4) -> float:
        """
        Time step at which the explicit scheme would be marginally stable.

        Backward Euler is unconditionally stable so this is not a constraint,
        but it is the natural accuracy scale for resolving conduction across
        one cell and is used to pick default time steps.
        """
        return safety * self.dz ** 2 / self.alpha

    def penetration_depth(self, duration: float) -> float:
        """Diffusion length sqrt(alpha * t) [m] over `duration`."""
        return float(np.sqrt(self.alpha * duration))

    def uniform(self, T: float) -> np.ndarray:
        return np.full(self.N, float(T))


def make_tape(thickness: float = 130e-6,
              N: int = 10,
              material_name: str = 'apc2_cf_peek',
              name: str = 'tape') -> Layer:
    """Incoming tape ply (130 um APC-2 by default)."""
    return Layer(name=name, thickness=thickness, N=N, **material(material_name))


def make_substrate(thickness: float = 2e-3,
                   N: int = 40,
                   material_name: str = 'apc2_cf_peek',
                   name: str = 'substrate') -> Layer:
    """
    Previously laid laminate below the nip-point.

    N = 40 over 2 mm gives 51 um cells. Laser heating is strongly
    surface-concentrated -- the diffusion length over a 0.3 s approach is only
    ~380 um -- so a coarser grid under-resolves the surface gradient and
    overpredicts the nip temperature by several kelvin.
    """
    return Layer(name=name, thickness=thickness, N=N, **material(material_name))


def make_roller(contact_time: float,
                radius: float = 0.040,
                N: int = 8,
                material_name: str = 'silicone_roller',
                skin_factor: float = 4.0,
                name: str = 'roller') -> Layer:
    """
    Compaction roller, modelled as a thin thermal *skin* rather than the whole
    cylinder.

    Only a shallow shell of the roller participates over a nip passage: for a
    silicone roller and a ~30 ms contact time the diffusion length is tens of
    microns, so meshing the full 40 mm radius would waste nodes on material
    that never leaves its initial temperature. The skin is sized at
    `skin_factor` diffusion lengths, beyond which the temperature rise is
    negligible, and the roller core is represented by the convective
    boundary condition applied to the skin's outer face.
    """
    props = material(material_name)
    alpha = props['k'] / (props['rho'] * props['cp'])
    skin = skin_factor * np.sqrt(alpha * max(contact_time, 1e-9))
    return Layer(name=name, thickness=float(min(skin, radius)), N=N, **props)


# ---------------------------------------------------------------------------
# Interfaces and boundaries
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ThermalResistance:
    """
    Zero-thickness thermal contact resistance between two layers [m^2.K/W].

    R = 0 means perfect contact, represented by a very large but finite
    conductance so the assembled matrix stays well posed.
    """
    R: float = 0.0

    H_PERFECT = 1e9   # W/(m^2.K)

    @property
    def h_contact(self) -> float:
        return self.H_PERFECT if self.R <= 0.0 else 1.0 / self.R

    @classmethod
    def perfect(cls) -> "ThermalResistance":
        return cls(0.0)

    @classmethod
    def from_conductance(cls, h: float) -> "ThermalResistance":
        return cls(1.0 / h if h > 0 else np.inf)


@dataclass
class SurfaceBC:
    """
    Convective + imposed-flux condition on one outer face of a stack:

        -k dT/dz = h (T_surface - T_ref) - Q(t)

    `Q` is either a constant or a callable of absolute time, and is positive
    into the domain.
    """
    h:     float = 0.0            # W/(m^2.K)
    T_ref: float = 20.0           # deg C
    Q:     float | Callable[[float], float] = 0.0   # W/m^2

    def flux(self, t: float) -> float:
        if callable(self.Q):
            return float(self.Q(t))
        return float(self.Q)

    @classmethod
    def adiabatic(cls) -> "SurfaceBC":
        return cls(h=0.0, T_ref=0.0, Q=0.0)


# ---------------------------------------------------------------------------
# Layer stack solver
# ---------------------------------------------------------------------------

def _solve_tridiagonal(ab: np.ndarray, b: np.ndarray) -> np.ndarray:
    if HAS_SCIPY:
        return solve_banded((1, 1), ab, b)
    n = len(b)
    A = np.zeros((n, n))
    np.fill_diagonal(A, ab[1])
    np.fill_diagonal(A[:, 1:], ab[0, 1:])
    np.fill_diagonal(A[1:, :], ab[2, :-1])
    return np.linalg.solve(A, b)


class LayerStack:
    """
    Chain of `Layer`s joined by `ThermalResistance`s, solved implicitly in 1D.

    Global node numbering runs from the top face of the first layer to the
    bottom face of the last. Layers do not share nodes: layer `j`'s bottom
    node and layer `j+1`'s top node are adjacent global indices coupled by
    `interfaces[j]`.
    """

    def __init__(self,
                 layers:     Sequence[Layer],
                 interfaces: Optional[Sequence[ThermalResistance]] = None):
        self.layers = list(layers)
        if not self.layers:
            raise ValueError("LayerStack needs at least one layer")

        n_gap = len(self.layers) - 1
        if interfaces is None:
            interfaces = [ThermalResistance.perfect()] * n_gap
        self.interfaces = list(interfaces)
        if len(self.interfaces) != n_gap:
            raise ValueError(
                f"{len(self.layers)} layers need {n_gap} interfaces, "
                f"got {len(self.interfaces)}"
            )

        counts = [L.N for L in self.layers]
        self.N = int(sum(counts))
        self._starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(int)

        z_parts, z0 = [], 0.0
        for L in self.layers:
            z_parts.append(np.linspace(z0, z0 + L.thickness, L.N))
            z0 += L.thickness
        self.z = np.concatenate(z_parts)
        self.total_thickness = z0

        self._build_coefficients()
        self._ab_cache: dict[float, np.ndarray] = {}

    # -- geometry helpers --------------------------------------------------

    def layer_index(self, name: str) -> int:
        for j, L in enumerate(self.layers):
            if L.name == name:
                return j
        raise KeyError(f"no layer named {name!r} in {[L.name for L in self.layers]}")

    def layer_slice(self, layer: int | str) -> slice:
        j = self.layer_index(layer) if isinstance(layer, str) else layer
        return slice(int(self._starts[j]), int(self._starts[j]) + self.layers[j].N)

    def interface_nodes(self, gap: int) -> tuple[int, int]:
        """Global indices (above, below) of the node pair bracketing `gap`."""
        upper = self.layer_slice(gap)
        return upper.stop - 1, upper.stop

    def diffusion_dt(self, safety: float = 0.4) -> float:
        return min(L.diffusion_dt(safety) for L in self.layers)

    # -- assembly ----------------------------------------------------------

    def _build_coefficients(self):
        """Per-node capacitance C [J/(m^2.K)] and internode conductance G."""
        self._C = np.zeros(self.N)
        self._G = np.zeros(max(self.N - 1, 0))

        for j, L in enumerate(self.layers):
            sl = self.layer_slice(j)
            self._C[sl] = L.rho_cp * L.dz
            # Face nodes own half a cell, so the layer total is rho*cp*thickness.
            self._C[sl.start] = L.rho_cp * L.dz / 2.0
            self._C[sl.stop - 1] = L.rho_cp * L.dz / 2.0
            self._G[sl.start:sl.stop - 1] = L.k / L.dz

        for gap, R in enumerate(self.interfaces):
            above, _ = self.interface_nodes(gap)
            self._G[above] = R.h_contact

    def _banded(self, dt: float, h_top: float, h_bot: float) -> np.ndarray:
        G = self._G
        diag = self._C / dt
        diag[1:] += G
        diag[:-1] += G
        diag[0] += h_top
        diag[-1] += h_bot

        ab = np.zeros((3, self.N))
        ab[1] = diag
        ab[0, 1:] = -G
        ab[2, :-1] = -G
        return ab

    def _banded_cached(self, dt, h_top, h_bot) -> np.ndarray:
        key = (round(dt, 15), h_top, h_bot)
        ab = self._ab_cache.get(key)
        if ab is None:
            ab = self._banded(dt, h_top, h_bot)
            self._ab_cache[key] = ab
        return ab

    # -- time integration --------------------------------------------------

    def solve(self,
              T_init:  np.ndarray,
              t_array: np.ndarray,
              top:     SurfaceBC,
              bottom:  SurfaceBC) -> np.ndarray:
        """
        Integrate from `t_array[0]` to `t_array[-1]`.

        `t_array` carries absolute times: the boundary flux callables are
        evaluated at those values, so a phase starting part-way through a
        process sees the correct portion of a time-dependent laser flux.

        Returns T_history with shape (len(t_array), N).
        """
        T = np.asarray(T_init, dtype=float).copy()
        if T.shape != (self.N,):
            raise ValueError(
                f"T_init has shape {T.shape}, expected ({self.N},)"
            )

        T_hist = np.empty((len(t_array), self.N))
        T_hist[0] = T

        for step in range(1, len(t_array)):
            dt = float(t_array[step] - t_array[step - 1])
            t = float(t_array[step])
            ab = self._banded_cached(dt, top.h, bottom.h)

            b = self._C / dt * T
            b[0] += top.h * top.T_ref + top.flux(t)
            b[-1] += bottom.h * bottom.T_ref + bottom.flux(t)

            T = _solve_tridiagonal(ab, b)
            T_hist[step] = T

        return T_hist

    # -- diagnostics -------------------------------------------------------

    def stored_energy(self, T: np.ndarray, T_ref: float = 0.0) -> float:
        """Sensible energy per unit area above `T_ref` [J/m^2]."""
        return float(np.sum(self._C * (np.asarray(T) - T_ref)))

    def interface_jump(self, T: np.ndarray, gap: int = 0) -> float:
        """Temperature discontinuity across `gap` [K], above minus below."""
        above, below = self.interface_nodes(gap)
        return float(T[above] - T[below])

    def __repr__(self):
        spec = " | ".join(f"{L.name}({L.thickness*1e6:.0f}um, N={L.N})"
                          for L in self.layers)
        return (f"LayerStack[{spec}]  total={self.total_thickness*1e3:.3f}mm, "
                f"N={self.N}")
