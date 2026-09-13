"""
Cooling Phase: Free Cooling of the Tape/Substrate Stack After Roller Lift-Off
=============================================================================

Third and last phase of the LATP thermal history. Once the material point has
travelled past the roller contact patch, the roller lifts off and the newly
consolidated tape is exposed to still air:

    tape top face    : convection to ambient (h_air)
    bond line        : the same contact resistance R_tc as consolidation
    substrate bottom : conduction into the tooling (h_tool)

There is no laser flux and no roller. The stack simply relaxes: the steep
through-thickness gradient left by the nip passage flattens out within a few
milliseconds (the tape is only 130 um thick), after which the whole stack
decays towards ambient on the much slower substrate timescale.

WHY THIS PHASE MATTERS
======================
PEEK crystallinity, and therefore the mechanical properties of the bond, is
set by the *rate* at which the interface passes down through the
crystallisation window, not by the peak temperature reached under the laser.
`CoolingSolver` therefore exposes cooling-rate and residence-time diagnostics
alongside the raw temperature history.

DISCRETISATION CONTINUITY
=========================
`from_consolidation` reuses the exact `Layer` objects from the consolidation
solver, so the profile handed over needs no interpolation -- only the roller
nodes are dropped. Rebuilding the tape or substrate with a different node
count between phases would silently smear the bond-line gradient, which is the
feature of interest.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from thermal_layers import (
    Layer,
    LayerStack,
    SurfaceBC,
    ThermalResistance,
    make_substrate,
    make_tape,
)

# Re-exported so a caller can build a cooling problem from this module alone.
__all__ = [
    'CoolingBC', 'CoolingSolver',
    'Layer', 'ThermalResistance', 'make_tape', 'make_substrate',
    'T_PEEK_MELT', 'T_PEEK_CRYST', 'T_PEEK_GLASS',
]


# APC-2 / PEEK transition temperatures [deg C]
T_PEEK_MELT = 343.0    # melting point
T_PEEK_CRYST = 300.0   # crystallisation onset on cooling
T_PEEK_GLASS = 143.0   # glass transition


@dataclass
class CoolingBC:
    """
    Boundary conditions after roller lift-off.

    h_air  : tape top face to ambient air [W/(m^2.K)]
    T_amb  : ambient temperature [deg C]
    h_tool : substrate bottom face to tooling [W/(m^2.K)]
    T_tool : tooling temperature [deg C]; defaults to T_amb
    """
    h_air:  float = 25.0
    T_amb:  float = 20.0
    h_tool: float = 1000.0
    T_tool: Optional[float] = None

    def tool_temperature(self) -> float:
        return self.T_amb if self.T_tool is None else self.T_tool


class CoolingSolver:
    """
    Tape over substrate, cooling freely with no roller and no laser.

    Parameters
    ----------
    tape, substrate : Layer
    interface : ThermalResistance
        Bond-line resistance; should match the consolidation phase.
    bc : CoolingBC
    """

    def __init__(self,
                 tape:      Layer,
                 substrate: Layer,
                 interface: ThermalResistance,
                 bc:        Optional[CoolingBC] = None):

        self.tape = tape
        self.substrate = substrate
        self.interface = interface
        self.bc = CoolingBC() if bc is None else bc

        self.stack = LayerStack([tape, substrate], [interface])
        self.z = self.stack.z
        self.N = self.stack.N

        self.tape_slice = self.stack.layer_slice(tape.name)
        self.substrate_slice = self.stack.layer_slice(substrate.name)
        self.i_tape_bottom, self.i_substrate_top = self.stack.interface_nodes(0)

    # -- construction ------------------------------------------------------

    @classmethod
    def from_consolidation(cls,
                           consolidation_solver,
                           bc: Optional[CoolingBC] = None) -> "CoolingSolver":
        """
        Build a matching cooling problem from a `StackedThermalSolver`.

        Shares the tape/substrate `Layer` objects and the bond resistance, so
        `StackedThermalSolver.tape_substrate_profile` output drops straight in
        as the initial condition.
        """
        return cls(
            tape=consolidation_solver.tape,
            substrate=consolidation_solver.substrate,
            interface=consolidation_solver.interface.as_resistance(),
            bc=bc,
        )

    # -- boundary conditions ----------------------------------------------

    def _top_bc(self) -> SurfaceBC:
        return SurfaceBC(h=self.bc.h_air, T_ref=self.bc.T_amb)

    def _bottom_bc(self) -> SurfaceBC:
        return SurfaceBC(h=self.bc.h_tool, T_ref=self.bc.tool_temperature())

    def initial_state(self,
                      T_tape:      np.ndarray | float,
                      T_substrate: np.ndarray | float) -> np.ndarray:
        """Assemble a global profile from per-layer data (stack order)."""
        T = np.empty(self.N)
        T[self.tape_slice] = np.broadcast_to(
            np.asarray(T_tape, dtype=float), (self.tape.N,))
        T[self.substrate_slice] = np.broadcast_to(
            np.asarray(T_substrate, dtype=float), (self.substrate.N,))
        return T

    # -- integration -------------------------------------------------------

    def solve(self, T_init: np.ndarray, t_array: np.ndarray) -> np.ndarray:
        """Integrate free cooling over absolute times `t_array`."""
        return self.stack.solve(T_init, t_array,
                                top=self._top_bc(),
                                bottom=self._bottom_bc())

    # -- diagnostics -------------------------------------------------------

    def bond_temperature(self, T_history: np.ndarray) -> np.ndarray:
        """
        Bond-line temperature [deg C], averaged across the contact resistance.

        The two nodes either side of the interface differ by the resistive
        jump; their mean is the representative weld-line temperature.
        """
        T = np.asarray(T_history)
        return 0.5 * (T[..., self.i_tape_bottom] + T[..., self.i_substrate_top])

    def cooling_rate(self, T_history: np.ndarray, t_array: np.ndarray,
                     node: Optional[int] = None) -> np.ndarray:
        """
        dT/dt [deg C/s] at `node` (default: the bond line). Negative while
        cooling.
        """
        T = (self.bond_temperature(T_history) if node is None
             else np.asarray(T_history)[..., node])
        return np.gradient(T, t_array)

    def crystallisation_rate(self, T_history: np.ndarray,
                             t_array: np.ndarray) -> float:
        """
        Bond-line cooling rate [deg C/s] at the crystallisation onset crossing.

        Returns NaN if the bond line never cools through T_PEEK_CRYST during
        the simulated window, which is the signal that the crystallisation
        history is not yet resolved and `t_cool` should be extended.
        """
        T_bond = self.bond_temperature(T_history)
        below = np.flatnonzero(T_bond <= T_PEEK_CRYST)
        if below.size == 0 or T_bond[0] <= T_PEEK_CRYST:
            return float('nan')
        idx = int(below[0])
        return float(np.gradient(T_bond, t_array)[idx])

    def time_above(self, T_history: np.ndarray, t_array: np.ndarray,
                   T_threshold: float) -> float:
        """Time [s] the bond line spends at or above `T_threshold`."""
        T_bond = self.bond_temperature(T_history)
        dt = np.diff(t_array, prepend=t_array[0])
        return float(np.sum(dt[T_bond >= T_threshold]))

    def __repr__(self):
        return (f"CoolingSolver({self.stack!r}, "
                f"R_tc={self.interface.R:.2e} m2K/W, "
                f"h_air={self.bc.h_air:g})")
