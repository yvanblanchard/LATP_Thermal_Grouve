"""
Per-Position Thermal History: T(x, z, t)
=========================================

Top-level driver that stitches the three LATP phase models into one continuous
thermal history for a Lagrangian material point on the substrate.

Each substrate position accumulates its history through three sequential
phases, each owned by its own module:

    Phase 1  HEATING        laser flux on the substrate surface
                            -> thermal_layers.LayerStack (substrate alone)
    Phase 2  CONSOLIDATION  roller in contact, tape pressed onto substrate
                            -> consolidation_thermal.StackedThermalSolver
    Phase 3  COOLING        roller lifted off, stack cools freely
                            -> cooling_thermal.CoolingSolver

The incoming tape is heated on a separate trajectory (it rides the roller, not
the substrate) and enters the history as an initial condition at the nip; see
`simulate_tape_at_nip`.

Heat flux distributions come from the 2D optical model via
`thermal_model_Grouve.compute_heat_fluxes_from_ray_tracing`, so the optical and
thermal models share one source of truth.


TIME / POSITION TRANSFORM
=========================
The flux field phi(x) is fixed in the frame of the nip-point; the material
moves through it. A point at arc-length x_i at t = 0 is at x_i + v*t at time t,
so the flux it sees is phi(x_i + v*t) and the timeline is:

    t = 0           point sits at x_i, at ambient
    t = t_nip       point reaches the nip-point, t_nip = |x_i| / v
    t = t_nip + t_c point leaves the roller, t_c = L_contact / v
    t = t_end       end of the cooling window

WHAT THE POSITION SWEEP REPRESENTS
==================================
phi(x) is zero outside the optical model's domain, so a point starting at
x_i = -5 mm only ever sees the last 5 mm of the flux field, while one starting
at the upstream edge sees all of it. The sweep is therefore not a set of
alternative steady states: it is the *course-start transient*. The first
material to pass under the nip at the beginning of a course has not travelled
the full heated zone and arrives cold; points progressively further upstream
have had progressively more exposure. The most-upstream position is the
fully-developed steady-state case, and comparing it against the near-nip
positions quantifies how much of the course is laid down under-heated.

DATA STRUCTURE
==============
PositionHistory  - (t_abs, T_surface, T_profile) for one x_i
SubstrateHistory - collection of PositionHistory for all x positions
"""

import time as _time
from dataclasses import dataclass, field
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np

from thermal_layers import (
    Layer,
    LayerStack,
    SurfaceBC,
    ThermalResistance,
    make_roller,
    make_substrate,
    make_tape,
)
from consolidation_thermal import (
    ConsolidationBC,
    InterfaceProps,
    StackedThermalSolver,
    orient_tape_for_stack,
)
from cooling_thermal import (
    CoolingBC,
    CoolingSolver,
    T_PEEK_CRYST,
    T_PEEK_GLASS,
    T_PEEK_MELT,
)
from thermal_model_Grouve import (
    BOUNDARY_CONDITIONS,
    PROCESS_PARAMS,
    compute_heat_fluxes_from_ray_tracing,
    create_heat_flux_interpolator,
)

def default_consolidation_bc() -> ConsolidationBC:
    """
    Consolidation BCs seeded from thermal_model_Grouve.BOUNDARY_CONDITIONS.

    Keeps the ambient and tooling values in step with the heating-phase model
    instead of repeating them; the roller parameters have no counterpart there
    and fall back to the ConsolidationBC defaults.
    """
    return ConsolidationBC(
        T_amb=float(BOUNDARY_CONDITIONS['T_far']),
        h_tool=float(BOUNDARY_CONDITIONS['htool']),
    )


def default_cooling_bc() -> CoolingBC:
    """Cooling BCs seeded from thermal_model_Grouve.BOUNDARY_CONDITIONS."""
    return CoolingBC(
        h_air=float(BOUNDARY_CONDITIONS['h']),
        T_amb=float(BOUNDARY_CONDITIONS['T_far']),
        h_tool=float(BOUNDARY_CONDITIONS['htool']),
    )


# ---------------------------------------------------------------------------
# Optical model access
# ---------------------------------------------------------------------------

_FLUX_CACHE: dict[float, tuple] = {}


def ray_traced_fluxes(laser_angle_deg: float) -> tuple:
    """
    Cached wrapper around the ray tracer.

    Tracing 5000 rays costs seconds, and the result depends only on the laser
    angle, so a position sweep must not re-run it per position.
    """
    key = float(laser_angle_deg)
    if key not in _FLUX_CACHE:
        _FLUX_CACHE[key] = compute_heat_fluxes_from_ray_tracing(key)
    return _FLUX_CACHE[key]


def max_flux_density(laser_power: Optional[float] = None) -> float:
    """
    Peak physical flux q_max = P_laser / A_beam  [W/m^2].

    Normalised optical flux is multiplied by this to become a thermal
    boundary condition, exactly as in thermal_model_Grouve.
    """
    power = PROCESS_PARAMS['laser_power'] if laser_power is None else laser_power
    beam_area = PROCESS_PARAMS['beam_width'] * PROCESS_PARAMS['beam_length']
    return power / beam_area


# ---------------------------------------------------------------------------
# Time grid helper
# ---------------------------------------------------------------------------

def _phase_times(t_start: float, duration: float, dt: float) -> np.ndarray:
    """
    Uniform grid over [t_start, t_start + duration] with spacing <= dt.

    linspace rather than arange so the phase boundary lands exactly on the
    requested time; phases are stitched at those boundaries and accumulated
    arange error would misalign them.
    """
    if duration <= 0.0:
        return np.array([t_start])
    n = max(2, int(np.ceil(duration / dt)) + 1)
    return t_start + np.linspace(0.0, duration, n)


# ---------------------------------------------------------------------------
# Per-position data container
# ---------------------------------------------------------------------------

@dataclass
class PositionHistory:
    """
    Full thermal history for one material point at arc-length x_i.

    Attributes
    ----------
    x            : arc-length position at t = 0 [m] (negative = upstream of nip)
    t_abs        : (N_t,) absolute time array [s]
    T_surface    : (N_t,) exposed surface temperature [deg C] - substrate top
                   during heating, tape top once the tape is laid
    T_profile    : (N_t, N_z) through-thickness field [deg C], tape nodes first
                   then substrate nodes. Tape nodes are NaN before the nip,
                   where the tape is not yet part of this stack.
    z_coords     : (N_z,) depth coordinates [m], 0 at the tape top face
    n_tape       : number of leading columns of T_profile belonging to the tape
    i_bond       : (tape-side, substrate-side) column indices of the bond line
    phase_times  : absolute times of the phase boundaries
    """
    x:           float
    t_abs:       np.ndarray
    T_surface:   np.ndarray
    T_profile:   np.ndarray
    z_coords:    np.ndarray
    n_tape:      int
    i_bond:      tuple[int, int]
    phase_times: dict = field(default_factory=dict)

    # -- convenience accessors --------------------------------------------

    def T_at_depth(self, z_target: float) -> np.ndarray:
        """Temperature history at the node nearest depth `z_target` [m]."""
        idx = int(np.argmin(np.abs(self.z_coords - z_target)))
        return self.T_profile[:, idx]

    def T_bond(self) -> np.ndarray:
        """
        Bond-line temperature history [deg C], mean of the two nodes either
        side of the contact resistance. NaN before the nip.
        """
        i_tape, i_sub = self.i_bond
        return 0.5 * (self.T_profile[:, i_tape] + self.T_profile[:, i_sub])

    def bond_delta_T(self) -> np.ndarray:
        """Temperature jump across the bond line [K]. NaN before the nip."""
        i_tape, i_sub = self.i_bond
        return self.T_profile[:, i_tape] - self.T_profile[:, i_sub]

    def peak_temperature(self) -> float:
        return float(np.nanmax(self.T_surface))

    def nip_temperature(self) -> float:
        """Surface temperature at nip-point passage [deg C]."""
        idx = int(np.searchsorted(self.t_abs, self.phase_times['heating_end']))
        idx = min(idx, len(self.t_abs) - 1)
        return float(self.T_surface[idx])

    def time_above(self, T_threshold: float) -> float:
        """Time [s] the surface spends at or above `T_threshold`."""
        dt = np.diff(self.t_abs, prepend=self.t_abs[0])
        hot = np.nan_to_num(self.T_surface, nan=-np.inf) >= T_threshold
        return float(np.sum(dt[hot]))

    def bond_time_above(self, T_threshold: float) -> float:
        """Time [s] the bond line spends at or above `T_threshold`."""
        dt = np.diff(self.t_abs, prepend=self.t_abs[0])
        hot = np.nan_to_num(self.T_bond(), nan=-np.inf) >= T_threshold
        return float(np.sum(dt[hot]))

    def cooling_rate_at(self, t_query: float) -> float:
        """Surface cooling rate dT/dt [deg C/s] at absolute time `t_query`."""
        return float(np.interp(t_query, self.t_abs,
                               np.gradient(self.T_surface, self.t_abs)))

    def crystallisation_cooling_rate(self) -> float:
        """
        Bond-line cooling rate [deg C/s] at the last downward crossing of the
        crystallisation onset -- the crossing that actually sets crystallinity.

        NaN if the bond line never reaches the onset, or has not cooled back
        below it by the end of the window (the signal to extend `t_cool`).
        """
        # The bond line is undefined before the nip, so the gradient is taken
        # over the finite part only; feeding NaNs to np.gradient poisons its
        # whole output.
        T_bond = self.T_bond()
        finite = np.flatnonzero(np.isfinite(T_bond))
        if finite.size < 2:
            return float('nan')
        T = T_bond[finite]
        t = self.t_abs[finite]

        hot = np.flatnonzero(T >= T_PEEK_CRYST)
        if hot.size == 0 or hot[-1] == len(T) - 1:
            return float('nan')
        return float(np.gradient(T, t)[hot[-1]])

    def max_through_thickness_gradient(self) -> np.ndarray:
        """
        Max |dT/dz| [deg C/m] at each time step, an indicator of thermal
        stress build-up. NaN tape nodes are ignored.
        """
        dz = np.diff(self.z_coords)
        with np.errstate(invalid='ignore', divide='ignore'):
            dTdz = np.diff(self.T_profile, axis=1) / dz[np.newaxis, :]
        # The tape/substrate interface is a zero-thickness resistance: dz = 0
        # there, so that column is not a gradient and must be dropped.
        dTdz[:, dz == 0.0] = np.nan
        return np.nanmax(np.abs(dTdz), axis=1)

    def __repr__(self):
        return (f"PositionHistory(x={self.x*1000:+.1f}mm, "
                f"T_peak={self.peak_temperature():.1f}C, "
                f"t_span={self.t_abs[-1]:.3f}s, "
                f"n_steps={len(self.t_abs)})")


# ---------------------------------------------------------------------------
# Collection of all positions
# ---------------------------------------------------------------------------

class SubstrateHistory:
    """
    Ordered collection of PositionHistory objects.

    Provides spatial maps (T_peak(x), t_above_melt(x), ...) and space-time
    matrices for contour plotting.
    """

    def __init__(self, positions: list[PositionHistory]):
        self.positions = sorted(positions, key=lambda p: p.x)
        self.x_arr = np.array([p.x for p in self.positions])

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return self.positions[idx]

    # -- spatial maps ------------------------------------------------------

    def peak_temperature_map(self) -> np.ndarray:
        return np.array([p.peak_temperature() for p in self.positions])

    def nip_temperature_map(self) -> np.ndarray:
        return np.array([p.nip_temperature() for p in self.positions])

    def time_above_map(self, T_threshold: float) -> np.ndarray:
        return np.array([p.time_above(T_threshold) for p in self.positions])

    def bond_time_above_map(self, T_threshold: float) -> np.ndarray:
        return np.array([p.bond_time_above(T_threshold) for p in self.positions])

    def bond_delta_T_map(self) -> np.ndarray:
        """Peak bond-line temperature jump at each position [K]."""
        return np.array([np.nanmax(np.abs(p.bond_delta_T()))
                         for p in self.positions])

    # -- space-time matrix -------------------------------------------------

    def build_xt_matrix(self, t_common: np.ndarray,
                        depth_idx: int = 0) -> np.ndarray:
        """
        Interpolate T(x_i, t) onto a common time grid.

        Returns T_xt with shape (N_x, len(t_common)).
        """
        T_xt = np.zeros((len(self.positions), len(t_common)))
        for i, p in enumerate(self.positions):
            col = p.T_profile[:, depth_idx]
            T_xt[i] = np.interp(t_common, p.t_abs, col,
                                left=col[0], right=col[-1])
        return T_xt


# ---------------------------------------------------------------------------
# Tape heating (separate trajectory, shared by all substrate positions)
# ---------------------------------------------------------------------------

def simulate_tape_at_nip(tape:             Layer,
                         velocity:         float,
                         flux_interp_tape: Callable[[float], float],
                         q_max:            float,
                         x_start:          float,
                         bc:               ConsolidationBC,
                         h_air:            float = 25.0,
                         dt:               Optional[float] = None,
                         ) -> np.ndarray:
    """
    Heat the incoming tape along its own trajectory and return its
    through-thickness profile at the nip-point.

    The tape rides the roller, so its exposure does not depend on the
    substrate position being tracked: this is solved once per velocity and
    reused across the whole position sweep.

    Node ordering of the returned profile is *heating order*: node 0 is the
    irradiated face. Pass it through `orient_tape_for_stack` before handing it
    to the consolidation solver, because the irradiated face becomes the bond
    face at the nip.

    Boundary conditions: the irradiated face sees ambient air, and the back
    face sits against the roller, so it conducts through the roller contact
    resistance towards the roller bulk rather than convecting to ambient.
    """
    duration = abs(x_start) / velocity
    dt = min(1e-3, duration / 200.0) if dt is None else dt
    t_arr = _phase_times(0.0, duration, dt)

    def Q(t: float) -> float:
        if t < 0.0 or t > duration:
            return 0.0
        x = min(x_start + velocity * t, 0.0)
        return max(0.0, float(flux_interp_tape(x))) * q_max

    stack = LayerStack([tape])
    T_hist = stack.solve(
        tape.uniform(bc.T_amb), t_arr,
        top=SurfaceBC(h=h_air, T_ref=bc.T_amb, Q=Q),
        bottom=SurfaceBC(h=ThermalResistance(bc.R_roller).h_contact,
                         T_ref=bc.T_roller_core),
    )
    return T_hist[-1]


# ---------------------------------------------------------------------------
# Per-position pipeline
# ---------------------------------------------------------------------------

class PositionPipeline:
    """
    Runs the full three-phase thermal history for one material position x_i.

    Parameters
    ----------
    x_position      : arc-length position at t = 0 [m], must be <= 0
    velocity        : placement velocity [m/s]
    flux_interp_sub : callable(x [m]) -> normalised substrate flux in [0, 1]
    q_max           : P_laser / A_beam [W/m^2]
    tape, substrate : Layer objects, shared across all three phases
    interface       : bond-line contact resistance
    consolidation_bc, cooling_bc : per-phase boundary data
    roller_contact_length : arc length of roller/substrate contact [m]
    t_cool          : cooling window after roller lift-off [s]
    h_air           : convection on the exposed substrate surface during
                      heating [W/(m^2.K)]
    """

    def __init__(self,
                 x_position:            float,
                 velocity:              float,
                 flux_interp_sub:       Callable[[float], float],
                 q_max:                 float,
                 tape:                  Layer,
                 substrate:             Layer,
                 interface:             InterfaceProps,
                 consolidation_bc:      ConsolidationBC,
                 cooling_bc:            CoolingBC,
                 roller:                Optional[Layer] = None,
                 roller_contact_length: float = 0.010,
                 t_cool:                float = 3.0,
                 h_air:                 float = 25.0):

        if x_position > 0.0:
            raise ValueError(
                f"x_position must be <= 0 (upstream of the nip), got {x_position}"
            )

        self.x_i = x_position
        self.v = velocity
        self.flux_f = flux_interp_sub
        self.q_max = q_max
        self.tape = tape
        self.substrate = substrate
        self.interface = interface
        self.cons_bc = consolidation_bc
        self.cool_bc = cooling_bc
        self.L_c = roller_contact_length
        self.t_cool = t_cool
        self.h_air = h_air

        self.t_nip = abs(self.x_i) / self.v
        self.t_c_end = self.t_nip + self.L_c / self.v

        self.roller = (make_roller(contact_time=self.L_c / self.v)
                       if roller is None else roller)

        self.consolidation = StackedThermalSolver(
            tape=tape, substrate=substrate, interface=interface,
            bc=consolidation_bc, roller=self.roller,
        )
        self.cooling = CoolingSolver.from_consolidation(
            self.consolidation, bc=cooling_bc,
        )

    # -- heat flux seen by the tracked point ------------------------------

    def _Q_heating(self, t: float) -> float:
        """
        Physical flux [W/m^2] absorbed at the tracked point's surface.

        The flux field is fixed relative to the nip; the material moves
        through it, so at time t the point sits at x_i + v*t and samples the
        optical profile there. After the nip the roller shadows the surface.
        """
        if t < 0.0 or t > self.t_nip:
            return 0.0
        x = min(self.x_i + self.v * t, 0.0)
        return max(0.0, float(self.flux_f(x))) * self.q_max

    # -- phase 1: heating --------------------------------------------------

    def _run_heating(self, dt: float):
        t_arr = _phase_times(0.0, self.t_nip, dt)
        stack = LayerStack([self.substrate])
        T_hist = stack.solve(
            self.substrate.uniform(self.cool_bc.T_amb), t_arr,
            top=SurfaceBC(h=self.h_air, T_ref=self.cool_bc.T_amb,
                          Q=self._Q_heating),
            bottom=SurfaceBC(h=self.cons_bc.h_tool,
                             T_ref=self.cons_bc.tool_temperature()),
        )
        return t_arr, T_hist

    # -- phase 2: consolidation -------------------------------------------

    def _run_consolidation(self, T_sub_at_nip: np.ndarray,
                           T_tape_at_nip: np.ndarray, dt: float):
        t_arr = _phase_times(self.t_nip, self.L_c / self.v, dt)
        T_init = self.consolidation.initial_state(
            T_tape=orient_tape_for_stack(T_tape_at_nip),
            T_substrate=T_sub_at_nip,
        )
        return t_arr, self.consolidation.solve(T_init, t_arr)

    # -- phase 3: cooling --------------------------------------------------

    def _run_cooling(self, T_stack_at_liftoff: np.ndarray, dt: float):
        t_arr = _phase_times(self.t_c_end, self.t_cool, dt)
        T_init = self.consolidation.tape_substrate_profile(T_stack_at_liftoff)
        return t_arr, self.cooling.solve(T_init, t_arr)

    # -- main entry point --------------------------------------------------

    def run(self,
            T_tape_at_nip: np.ndarray | float,
            dt_heat:       Optional[float] = None,
            dt_consol:     Optional[float] = None,
            dt_cool:       Optional[float] = None) -> PositionHistory:
        """
        Run all three phases and return the stitched PositionHistory.

        `T_tape_at_nip` is the tape profile in heating order (node 0 =
        irradiated face), from `simulate_tape_at_nip`; a scalar is broadcast.
        """
        t_contact = self.L_c / self.v
        dt_heat = min(1e-3, max(self.t_nip, 1e-9) / 200.0) if dt_heat is None else dt_heat
        dt_consol = min(1e-4, t_contact / 100.0) if dt_consol is None else dt_consol
        dt_cool = min(5e-3, self.t_cool / 300.0) if dt_cool is None else dt_cool

        t_heat, T_heat = self._run_heating(dt_heat)
        T_sub_at_nip = T_heat[-1]

        if np.isscalar(T_tape_at_nip):
            T_tape_at_nip = self.tape.uniform(float(T_tape_at_nip))

        t_consol, T_consol = self._run_consolidation(
            T_sub_at_nip, T_tape_at_nip, dt_consol)

        t_cool, T_cool = self._run_cooling(T_consol[-1], dt_cool)

        # Stitch onto one absolute time axis, dropping the duplicated first
        # sample of each later phase.
        t_abs = np.concatenate([t_heat, t_consol[1:], t_cool[1:]])

        n_tape = self.tape.N
        n_z = n_tape + self.substrate.N

        T_heat_padded = np.full((len(t_heat), n_z), np.nan)
        T_heat_padded[:, n_tape:] = T_heat

        T_full = np.concatenate([
            T_heat_padded,
            self.consolidation.tape_substrate_profile(T_consol[1:]),
            T_cool[1:],
        ], axis=0)

        # The exposed surface is the substrate top until the tape covers it.
        T_surface = np.empty(len(t_abs))
        n1 = len(t_heat)
        T_surface[:n1] = T_full[:n1, n_tape]
        T_surface[n1:] = T_full[n1:, 0]

        return PositionHistory(
            x=self.x_i,
            t_abs=t_abs,
            T_surface=T_surface,
            T_profile=T_full,
            z_coords=self.cooling.z,
            n_tape=n_tape,
            i_bond=(self.cooling.i_tape_bottom, self.cooling.i_substrate_top),
            phase_times={
                'heating_end': self.t_nip,
                'consolidation_end': self.t_c_end,
            },
        )


# ---------------------------------------------------------------------------
# Multi-position driver
# ---------------------------------------------------------------------------

def build_substrate_history(x_positions:           np.ndarray,
                            velocity:              float,
                            laser_angle_deg:       float = 22.0,
                            laser_power:           Optional[float] = None,
                            tape:                  Optional[Layer] = None,
                            substrate:             Optional[Layer] = None,
                            R_tc:                  float = 1e-4,
                            consolidation_bc:      Optional[ConsolidationBC] = None,
                            cooling_bc:            Optional[CoolingBC] = None,
                            roller_contact_length: float = 0.010,
                            t_cool:                float = 3.0,
                            verbose:               bool = True,
                            ) -> SubstrateHistory:
    """
    Run the full three-phase pipeline for every x position.

    Heat fluxes are taken from the 2D optical model at `laser_angle_deg`. The
    tape heating solve and the ray trace are both done once and shared across
    positions.
    """
    tape = make_tape() if tape is None else tape
    substrate = make_substrate() if substrate is None else substrate
    cons_bc = default_consolidation_bc() if consolidation_bc is None else consolidation_bc
    cool_bc = default_cooling_bc() if cooling_bc is None else cooling_bc
    interface = InterfaceProps(R_tc=R_tc)

    sub_dist, sub_flux, tape_dist, tape_flux = ray_traced_fluxes(laser_angle_deg)
    flux_interp_sub = create_heat_flux_interpolator(sub_dist, sub_flux)
    flux_interp_tape = create_heat_flux_interpolator(tape_dist, tape_flux)
    q_max = max_flux_density(laser_power)

    roller = make_roller(contact_time=roller_contact_length / velocity)

    n = len(x_positions)
    t_start = _time.perf_counter()

    print(f"\n{'='*72}")
    print(f"  SUBSTRATE HISTORY - {n} positions  |  v = {velocity} m/s  "
          f"|  alpha = {laser_angle_deg} deg")
    print(f"  q_max = {q_max/1e5:.2f} W/mm^2   R_tc = {R_tc:.2e} m^2K/W")
    print(f"{'='*72}")

    T_tape_nip = simulate_tape_at_nip(
        tape=tape, velocity=velocity, flux_interp_tape=flux_interp_tape,
        q_max=q_max, x_start=float(np.min(tape_dist)), bc=cons_bc,
        h_air=cool_bc.h_air,
    )
    print(f"  Tape at nip: bond face {T_tape_nip[0]:.1f} C, "
          f"roller face {T_tape_nip[-1]:.1f} C")

    histories = []
    for i, x_i in enumerate(x_positions):
        if verbose:
            print(f"  [{i+1}/{n}]  x = {x_i*1000:+7.1f} mm", end='  ')

        pipeline = PositionPipeline(
            x_position=float(x_i),
            velocity=velocity,
            flux_interp_sub=flux_interp_sub,
            q_max=q_max,
            tape=tape,
            substrate=substrate,
            interface=interface,
            consolidation_bc=cons_bc,
            cooling_bc=cool_bc,
            roller=roller,
            roller_contact_length=roller_contact_length,
            t_cool=t_cool,
            h_air=cool_bc.h_air,
        )
        ph = pipeline.run(T_tape_at_nip=T_tape_nip)
        histories.append(ph)

        if verbose:
            print(f"T_nip={ph.nip_temperature():6.1f}C  "
                  f"T_bond_max={np.nanmax(ph.T_bond()):6.1f}C  "
                  f"t>melt={ph.bond_time_above(T_PEEK_MELT)*1000:6.2f}ms  "
                  f"n={len(ph.t_abs)}")

    elapsed = _time.perf_counter() - t_start
    print(f"\n  Done - {n} positions in {elapsed:.2f}s "
          f"({elapsed/n*1000:.1f} ms/position)")

    return SubstrateHistory(histories)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_position_histories(history: SubstrateHistory,
                            velocity: float,
                            laser_angle_deg: float = 22.0):
    """
    Four-panel figure:
      A - T_surface(t) for all positions, coloured by x
      B - Spatial maps: T_nip(x) and bond time above melt
      C - Space-time contour T(x, t) at the exposed surface
      D - Through-thickness T(z) snapshots for a representative position
    """
    positions = history.positions
    n = len(positions)
    x_mm = history.x_arr * 1000
    colours = plt.cm.viridis(np.linspace(0, 1, n))

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        f'Per-Position Thermal Histories - v = {velocity} m/s, '
        f'alpha = {laser_angle_deg} deg\n'
        f'Heating -> consolidation -> cooling  ({n} substrate positions)',
        fontsize=12,
    )

    # -- A: surface temperature histories ---------------------------------
    ax = axes[0, 0]
    for ph, clr in zip(positions, colours):
        ax.plot(ph.t_abs, ph.T_surface, color=clr, lw=1.2, alpha=0.85)
        ax.plot(ph.t_abs, ph.T_bond(), color=clr, lw=0.9, ls='--', alpha=0.6)

    ax.axhline(T_PEEK_MELT, color='orange', lw=1.5, ls='-.',
               label=f'Melt {T_PEEK_MELT:.0f}C')
    ax.axhline(T_PEEK_CRYST, color='purple', lw=1.2, ls=':',
               label=f'Cryst {T_PEEK_CRYST:.0f}C')
    ax.axhline(T_PEEK_GLASS, color='gray', lw=1.0, ls=':',
               label=f'Tg {T_PEEK_GLASS:.0f}C')
    sm = plt.cm.ScalarMappable(cmap='viridis',
                               norm=plt.Normalize(x_mm.min(), x_mm.max()))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='start position x (mm)')
    ax.set_xlabel('Absolute time (s)')
    ax.set_ylabel('Temperature (deg C)')
    # The step at each curve's nip time is physical, not numerical: the
    # exposed surface stops being the bare substrate and becomes the tape.
    ax.set_title('A - Exposed surface (solid) & bond line (dashed)\n'
                 'step at nip = tape covers the substrate')
    ax.set_xscale('symlog', linthresh=0.01)
    ax.set_xlim(left=0.0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # -- B: spatial maps ---------------------------------------------------
    ax = axes[0, 1]
    ax2 = ax.twinx()

    T_nip = history.nip_temperature_map()
    T_bond_peak = np.array([np.nanmax(p.T_bond()) for p in positions])
    t_melt = history.bond_time_above_map(T_PEEK_MELT) * 1000

    ax.plot(x_mm, T_nip, 'b-o', ms=4, lw=2, label='Substrate T at nip (C)')
    ax.plot(x_mm, T_bond_peak, 'g-^', ms=4, lw=2, label='Peak bond-line T (C)')
    ax2.plot(x_mm, t_melt, 'r-s', ms=4, lw=1.5, alpha=0.8,
             label='Bond time above melt (ms)')

    ax.axhline(T_PEEK_MELT, color='orange', lw=1.2, ls='-.', alpha=0.7,
               label=f'Melt {T_PEEK_MELT:.0f}C')
    ax.axhline(T_PEEK_CRYST, color='purple', lw=1.0, ls=':', alpha=0.7)
    ax.set_xlabel('Start position x (mm)   [-> less exposure ->]')
    ax.set_ylabel('Temperature (deg C)', color='blue')
    ax2.set_ylabel('Bond time above melt (ms)', color='red')
    ax.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='red')
    # An all-zero residence would otherwise autoscale to a meaningless
    # +/- epsilon range around zero.
    ax2.set_ylim(0, max(t_melt.max() * 1.1, 1.0))
    ax.set_title('B - Course-start transient: nip & bond-line temperatures')
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8)
    ax.grid(True, alpha=0.3)

    # -- C: space-time contour --------------------------------------------
    ax = axes[1, 0]
    t_max = max(ph.t_abs[-1] for ph in positions)
    t_common = np.linspace(0, t_max, 400)
    # Track the substrate top face: it is defined over the whole history,
    # whereas the tape nodes do not exist before the nip.
    T_xt = history.build_xt_matrix(t_common, depth_idx=positions[0].n_tape)

    cf = ax.contourf(t_common, x_mm, T_xt, levels=40, cmap='hot')
    plt.colorbar(cf, ax=ax, label='Temperature (deg C)')
    ax.contour(t_common, x_mm, T_xt,
               levels=[T_PEEK_GLASS, T_PEEK_CRYST, T_PEEK_MELT],
               colors=['cyan', 'magenta', 'white'], linewidths=1.2)
    ax.set_xlabel('Absolute time (s)')
    ax.set_ylabel('Start position x (mm)')
    ax.set_title('C - Space-time contour T(x, t) at substrate top face')
    ax.grid(False)

    # -- D: through-thickness snapshots -----------------------------------
    ax = axes[1, 1]
    ph = positions[n // 2]
    z_mm = ph.z_coords * 1000
    d_interface = z_mm[ph.n_tape - 1]

    ax.axhspan(z_mm[0], d_interface, alpha=0.10, color='blue')
    ax.axhspan(d_interface, z_mm[-1], alpha=0.08, color='red')
    ax.axhline(d_interface, color='purple', lw=1.5, ls='--',
               label='Tape/substrate bond line')

    snap_times = np.concatenate([
        [ph.phase_times['heating_end']],
        np.linspace(ph.phase_times['heating_end'],
                    ph.phase_times['consolidation_end'], 3)[1:],
        np.geomspace(ph.phase_times['consolidation_end'] * 1.05,
                     ph.t_abs[-1], 3),
    ])
    snap_colours = plt.cm.plasma(np.linspace(0.1, 0.9, len(snap_times)))

    for t_s, clr in zip(snap_times, snap_colours):
        idx = min(int(np.searchsorted(ph.t_abs, t_s)), len(ph.t_abs) - 1)
        T_snap = ph.T_profile[idx]
        valid = ~np.isnan(T_snap)
        if valid.any():
            ax.plot(T_snap[valid], z_mm[valid], color=clr, lw=1.8,
                    label=f't = {ph.t_abs[idx]:.3f}s')

    ax.axvline(T_PEEK_MELT, color='orange', lw=1, ls='-.', alpha=0.7)
    ax.axvline(T_PEEK_CRYST, color='purple', lw=1, ls=':', alpha=0.7)
    ax.invert_yaxis()
    ax.set_xlabel('Temperature (deg C)')
    ax.set_ylabel('Depth z from tape top (mm)')
    ax.set_title(f'D - Through-thickness T(z)\nstart x = {ph.x*1000:.1f} mm')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f'position_histories_v{velocity}_alpha{laser_angle_deg:g}deg.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"\n  Figure saved: {fname}")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    velocity = PROCESS_PARAMS['velocities'][1]
    laser_angle = PROCESS_PARAMS['laser_angles'][0]

    tape = make_tape(thickness=PROCESS_PARAMS['tape_thickness'])
    substrate = make_substrate(thickness=PROCESS_PARAMS['substrate_thickness'])

    sub_dist, _, _, _ = ray_traced_fluxes(laser_angle)
    x_upstream = float(np.min(sub_dist))

    # From the fully-developed case at the upstream edge of the flux field to
    # the cold-start case just ahead of the nip.
    x_positions = np.linspace(x_upstream, 0.05 * x_upstream, 12)

    history = build_substrate_history(
        x_positions=x_positions,
        velocity=velocity,
        laser_angle_deg=laser_angle,
        tape=tape,
        substrate=substrate,
        R_tc=1e-4,
        roller_contact_length=0.010,
        t_cool=3.0,
    )

    plot_position_histories(history, velocity, laser_angle)

    ph = history[-1]
    print(f"\nCold-start position x = {ph.x*1000:+.1f} mm:")
    print(f"  Surface T at nip        : {ph.nip_temperature():.1f} C")
    print(f"  Peak surface T          : {ph.peak_temperature():.1f} C")
    print(f"  Peak bond-line T        : {np.nanmax(ph.T_bond()):.1f} C")
    print(f"  Peak bond dT jump       : {np.nanmax(np.abs(ph.bond_delta_T())):.1f} K")
    print(f"  Bond time above melt    : {ph.bond_time_above(T_PEEK_MELT)*1000:.2f} ms")
    print(f"  Bond time above cryst   : {ph.bond_time_above(T_PEEK_CRYST)*1000:.2f} ms")
    print(f"  Cooling rate at cryst   : {ph.crystallisation_cooling_rate():.0f} C/s")

    ph0 = history[0]
    print(f"\nSteady-state position x = {ph0.x*1000:+.1f} mm:")
    print(f"  Surface T at nip        : {ph0.nip_temperature():.1f} C")
    print(f"  Peak bond-line T        : {np.nanmax(ph0.T_bond()):.1f} C")
    print(f"  Bond time above melt    : {ph0.bond_time_above(T_PEEK_MELT)*1000:.2f} ms")


if __name__ == '__main__':
    main()
