from __future__ import annotations

from dataclasses import dataclass


DEFAULT_VEHICLE_LENGTH_M = 5.0
DEFAULT_JAM_DENSITY_VPM = 1.0 / DEFAULT_VEHICLE_LENGTH_M


@dataclass(frozen=True)
class VehicleClass:
    """Vehicle-class parameters used by MIXSim.

    ``time_gap_s`` is the class-specific time gap used in the macroscopic
    spacing closure

        s(v) = v * tau + L_v,

    where ``L_v`` is the effective standstill spacing.  In the default manuscript
    configuration all classes use the same effective standstill spacing, so
    jam density is ``1 / L_v`` while the class-specific time gap changes the
    capacity and backward congested-wave speed.
    """

    name: str
    time_gap_s: float
    vehicle_length_m: float = DEFAULT_VEHICLE_LENGTH_M
    color: str = "#000000"
    reference_share: float = 0.5

    @property
    def implied_jam_density_vpm(self) -> float:
        return 1.0 / self.vehicle_length_m

    def __post_init__(self) -> None:
        if self.time_gap_s <= 0:
            raise ValueError("time_gap_s must be positive")
        if self.vehicle_length_m <= 0:
            raise ValueError("vehicle_length_m must be positive")
        if self.reference_share < 0:
            raise ValueError("reference_share cannot be negative")
