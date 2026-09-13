"""
Consolidation Phase: Roller / Tape / Substrate Under the Nip
=============================================================

Second phase of the LATP thermal history. The incoming tape, heated on its
bonding face by the laser, meets the substrate at the nip-point and is pressed
onto it by the compaction roller for the duration of the contact patch:

    t_contact = roller_contact_length / velocity

Three things distinguish this phase from free heating or free cooling, and all
three change the answer materially:

1.  NO LASER FLUX. The roller shadows the nip, so both outer faces are purely
    conductive/convective. The stack only redistributes energy already stored.

2.  THE ROLLER IS A HEAT SINK ON THE TAPE'S TOP FACE. The tape is 130 um
    thin, so replacing the roller with an air-convection condition
    overpredicts the tape temperature substantially: a silicone roller at
    ambient draws far more heat out of the tape than still air does. The
    roller is therefore modelled as an explicit thermal skin (see
    `thermal_layers.make_roller`) joined to the tape by its own contact
    resistance.

3.  THE BOND INTERFACE IS RESISTIVE. Tape and substrate are not in perfect
    contact at the nip: only a fraction of the nominal area is actually
    touching, the rest being air-filled valleys between asperities. That
    contact resistance R_tc sets the temperature *discontinuity* across the
    bond line, which is the quantity that governs whether the interface
    reaches healing temperature at all.

LAYER ORDER AND TAPE ORIENTATION
================================
The stack is assembled top-to-bottom as

    roller | tape | substrate

so global node 0 is the roller's outer face and the last node is the tooling
face of the substrate.

Note that the laser irradiates the face of the tape that will *become the bond
face* -- the tape wraps the roller and presents its future-bonding side to the
beam. The hottest tape node at the nip is therefore the one adjacent to the
substrate, not the one against the roller. Callers must hand
`initial_state` a tape profile already oriented top-to-bottom for this stack;
`orient_tape_for_stack` performs that flip.
"""

from dataclasses import dataclass
from typing import Optional

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


# ---------------------------------------------------------------------------
# Bond-line interface
# ---------------------------------------------------------------------------

@dataclass
class InterfaceProps:
    """
    Thermal contact resistance of the tape/substrate bond line [m^2.K/W].

    Use `R_tc` directly when it is being treated as a fitting parameter, or
    build it from a degree of intimate contact with `from_intimate_contact`
    when tying it to the consolidation state.
    """
    R_tc: float = 1e-4

    def as_resistance(self) -> ThermalResistance:
        return ThermalResistance(self.R_tc)

    @property
    def h_contact(self) -> float:
        return self.as_resistance().h_contact

    @classmethod
    def from_intimate_contact(cls,
                              D_ic:      float,
                              gap:       float = 10e-6,
                              k_polymer: float = 0.7,
                              k_air:     float = 0.030) -> "InterfaceProps":
        """
        Estimate R_tc from the degree of intimate contact D_ic in [0, 1].

        The interface is idealised as a layer of thickness `gap` (the asperity
        height scale) in which a fraction D_ic of the area conducts through
        polymer and the remainder through trapped air. Treating the two paths
        as parallel:

            h_c = D_ic * k_polymer/gap + (1 - D_ic) * k_air/gap
            R_tc = 1 / h_c

        D_ic = 1 recovers the resistance of a fully wetted interface,
        gap/k_polymer; D_ic = 0 gives the much larger air-gap resistance. The
        growth of D_ic under roller pressure is not modelled here.
        """
        if not 0.0 <= D_ic <= 1.0:
            raise ValueError(f"D_ic must be in [0, 1], got {D_ic}")
        h_c = (D_ic * k_polymer + (1.0 - D_ic) * k_air) / gap
        return cls(R_tc=1.0 / h_c)


# ---------------------------------------------------------------------------
# Boundary data
# ---------------------------------------------------------------------------

@dataclass
class ConsolidationBC:
    """
    Boundary conditions for the nip passage.

    T_amb         : ambient air temperature [deg C]
    h_tool        : substrate-to-tooling conductance [W/(m^2.K)]
    T_tool        : tooling temperature [deg C]; defaults to T_amb
    R_roller      : roller-to-tape contact resistance [m^2.K/W]
    T_roller_core : bulk roller temperature [deg C]
    h_roller_core : conductance from the modelled roller skin to that bulk
                    [W/(m^2.K)]. The roller rotates, so the skin is
                    continuously renewed towards the core temperature; a large
                    value pins the skin's outer face to T_roller_core, a small
                    one lets the skin float and behave adiabatically.
    """
    T_amb:         float = 20.0
    h_tool:        float = 1000.0
    T_tool:        Optional[float] = None
    R_roller:      float = 5e-4
    T_roller_core: float = 20.0
    h_roller_core: float = 2000.0

    def tool_temperature(self) -> float:
        return self.T_amb if self.T_tool is None else self.T_tool


# ---------------------------------------------------------------------------
# Tape orientation
# ---------------------------------------------------------------------------

def orient_tape_for_stack(T_tape_heating: np.ndarray) -> np.ndarray:
    """
    Convert a tape profile from heating-phase order to stack order.

    During heating the tape is solved as a standalone layer whose node 0 is
    the irradiated face. In the consolidation stack the tape sits under the
    roller with its irradiated face downwards, against the substrate, so the
    profile is reversed.
    """
    return np.asarray(T_tape_heating, dtype=float)[::-1].copy()


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class StackedThermalSolver:
    """
    Roller / tape / substrate stack during the nip passage.

    Parameters
    ----------
    tape, substrate : Layer
        Same objects used by the heating and cooling phases, so profiles can
        be handed between phases without interpolation.
    interface : InterfaceProps
        Bond-line resistance between tape and substrate.
    bc : ConsolidationBC
    roller : Layer, optional
        Roller skin. Pass None to omit the roller entirely and expose the tape
        top face to air at `bc.T_amb` with conductance `h_air` -- useful for
        isolating the roller's effect.
    h_air : float
        Only used when `roller` is None.
    """

    def __init__(self,
                 tape:      Layer,
                 substrate: Layer,
                 interface: InterfaceProps,
                 bc:        ConsolidationBC,
                 roller:    Optional[Layer] = None,
                 h_air:     float = 25.0):

        self.tape = tape
        self.substrate = substrate
        self.interface = interface
        self.bc = bc
        self.roller = roller
        self.h_air = h_air

        bond = interface.as_resistance()
        if roller is None:
            layers = [tape, substrate]
            gaps = [bond]
        else:
            layers = [roller, tape, substrate]
            gaps = [ThermalResistance(bc.R_roller), bond]

        self.stack = LayerStack(layers, gaps)
        self.z = self.stack.z
        self.N = self.stack.N

        self.tape_slice = self.stack.layer_slice(tape.name)
        self.substrate_slice = self.stack.layer_slice(substrate.name)
        self.roller_slice = (None if roller is None
                             else self.stack.layer_slice(roller.name))

        self.bond_gap = len(gaps) - 1
        self.i_tape_bottom, self.i_substrate_top = \
            self.stack.interface_nodes(self.bond_gap)

    # -- boundary conditions ----------------------------------------------

    def _top_bc(self) -> SurfaceBC:
        if self.roller is None:
            return SurfaceBC(h=self.h_air, T_ref=self.bc.T_amb)
        return SurfaceBC(h=self.bc.h_roller_core, T_ref=self.bc.T_roller_core)

    def _bottom_bc(self) -> SurfaceBC:
        return SurfaceBC(h=self.bc.h_tool, T_ref=self.bc.tool_temperature())

    # -- state assembly ----------------------------------------------------

    def initial_state(self,
                      T_tape:      np.ndarray | float,
                      T_substrate: np.ndarray | float,
                      T_roller:    Optional[np.ndarray | float] = None
                      ) -> np.ndarray:
        """
        Build a global initial profile from per-layer data.

        `T_tape` must already be in stack order (irradiated face last); see
        `orient_tape_for_stack`. Scalars are broadcast. `T_roller` defaults to
        `bc.T_roller_core`.
        """
        T = np.empty(self.N)
        T[self.tape_slice] = np.broadcast_to(
            np.asarray(T_tape, dtype=float), (self.tape.N,))
        T[self.substrate_slice] = np.broadcast_to(
            np.asarray(T_substrate, dtype=float), (self.substrate.N,))
        if self.roller is not None:
            fill = self.bc.T_roller_core if T_roller is None else T_roller
            T[self.roller_slice] = np.broadcast_to(
                np.asarray(fill, dtype=float), (self.roller.N,))
        return T

    def contact_time(self, roller_contact_length: float, velocity: float) -> float:
        return roller_contact_length / velocity

    # -- integration -------------------------------------------------------

    def solve(self, T_init: np.ndarray, t_array: np.ndarray) -> np.ndarray:
        """Integrate the nip passage over absolute times `t_array`."""
        return self.stack.solve(T_init, t_array,
                                top=self._top_bc(),
                                bottom=self._bottom_bc())

    # -- extraction --------------------------------------------------------

    def tape_substrate_profile(self, T: np.ndarray) -> np.ndarray:
        """
        Drop the roller nodes, returning the tape+substrate profile.

        This is the handover format for the cooling phase, whose stack has no
        roller layer.
        """
        T = np.asarray(T)
        if self.roller is None:
            return T[..., :].copy()
        return T[..., self.tape_slice.start:].copy()

    def bond_temperatures(self, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Temperatures either side of the bond line: (tape side, substrate side)."""
        T = np.asarray(T)
        return T[..., self.i_tape_bottom], T[..., self.i_substrate_top]

    def bond_delta_T(self, T: np.ndarray) -> np.ndarray:
        """Temperature jump across the bond line [K]."""
        tape_side, sub_side = self.bond_temperatures(T)
        return tape_side - sub_side

    def heat_into_roller(self, T: np.ndarray) -> np.ndarray:
        """
        Instantaneous flux from tape into the roller [W/m^2], positive when the
        roller is extracting heat.
        """
        if self.roller is None:
            return np.zeros(np.shape(T)[:-1])
        T = np.asarray(T)
        r_bottom, tape_top = self.stack.interface_nodes(0)
        h = ThermalResistance(self.bc.R_roller).h_contact
        return h * (T[..., tape_top] - T[..., r_bottom])

    def __repr__(self):
        return (f"StackedThermalSolver({self.stack!r}, "
                f"R_tc={self.interface.R_tc:.2e} m2K/W)")


# ---------------------------------------------------------------------------
# Convenience constructor
# ---------------------------------------------------------------------------

def make_consolidation_solver(velocity:              float,
                              roller_contact_length: float = 0.010,
                              tape:                  Optional[Layer] = None,
                              substrate:             Optional[Layer] = None,
                              R_tc:                  float = 1e-4,
                              bc:                    Optional[ConsolidationBC] = None,
                              with_roller:           bool = True,
                              roller_radius:         float = 0.040,
                              ) -> StackedThermalSolver:
    """
    Build a solver with the roller skin sized for this nip passage.

    The skin thickness depends on the contact time, hence on velocity, which
    is why the roller cannot be constructed independently of the process
    conditions.
    """
    tape = make_tape() if tape is None else tape
    substrate = make_substrate() if substrate is None else substrate
    bc = ConsolidationBC() if bc is None else bc

    t_contact = roller_contact_length / velocity
    roller = (make_roller(contact_time=t_contact, radius=roller_radius)
              if with_roller else None)

    return StackedThermalSolver(
        tape=tape,
        substrate=substrate,
        interface=InterfaceProps(R_tc=R_tc),
        bc=bc,
        roller=roller,
    )
