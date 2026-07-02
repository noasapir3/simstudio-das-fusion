"""Physical and simulation constants for SimStudio.

Centralising magic numbers here makes it easy to understand what each value
represents and to adjust simulation parameters in one place rather than hunting
through multiple files.

All values carry SI units unless the name contains a unit suffix (e.g. _KMH, _M).
"""

# ---------------------------------------------------------------------------
# Road / lane geometry
# ---------------------------------------------------------------------------

#: Standard urban lane width (m).
DEFAULT_LANE_WIDTH_M: float = 3.6

#: Default road speed limit used when a segment carries no annotation (~50 km/h).
DEFAULT_SPEED_LIMIT_MPS: float = 13.9

# ---------------------------------------------------------------------------
# Vehicle speed bounds
# ---------------------------------------------------------------------------

#: Minimum realistic urban vehicle speed (km/h).
SPEED_MIN_KMH: float = 20.0

#: Maximum modelled vehicle speed (km/h).
SPEED_MAX_KMH: float = 100.0

#: Hard upper limit on instantaneous vehicle speed (m/s).  ~150 km/h.
SPEED_HARD_CAP_MPS: float = 42.0

# ---------------------------------------------------------------------------
# Longitudinal dynamics
# ---------------------------------------------------------------------------

#: First-order speed-tracking time constant (s).
#: Lower = snappier acceleration response.
ACCEL_RESPONSE_TAU_S: float = 1.5

#: Acceleration command is zeroed when |Δv| is smaller than this (m/s).
ACCEL_CMD_DEADBAND_MPS: float = 0.12

#: Speed is considered "at target" (cruise hold) when |Δv| < this (m/s).
CRUISE_DEADBAND_MPS: float = 0.18

#: Fraction of the [speed_min, speed_max] range used as the default speed
#: standard-deviation when no per-vehicle std is specified.
SPEED_STD_RATIO: float = 0.18

#: Fallback maximum acceleration (m/s²) when no vehicle-level value is set.
DEFAULT_MAX_ACCEL_MPS2: float = 1.8

#: Fallback maximum deceleration (m/s²) when no vehicle-level value is set.
DEFAULT_MAX_DECEL_MPS2: float = 2.6

# ---------------------------------------------------------------------------
# Lateral dynamics  (Ornstein–Uhlenbeck lane-drift model)
# ---------------------------------------------------------------------------

#: OU process mean-reversion time constant for lateral lane drift (s).
LATERAL_DRIFT_TAU_S: float = 1.5

# ---------------------------------------------------------------------------
# Camera sensor noise model  (pinhole projection)
# ---------------------------------------------------------------------------

#: Reference image width in pixels used when converting angular error to metres.
CAMERA_PIXEL_WIDTH: int = 1280

#: Localisation noise standard deviation in pixels (before range scaling).
CAMERA_SIGMA_PX: float = 1.5

# ---------------------------------------------------------------------------
# Traffic density targets  (vehicles per km per lane)
# ---------------------------------------------------------------------------

#: Target vehicle densities indexed by Segment.traffic_level annotation.
TRAFFIC_DENSITY_VPK: dict = {
    "light":  10.0,
    "medium": 18.0,
    "heavy":  26.0,
}

# ---------------------------------------------------------------------------
# IDM car-following model  (Intelligent Driver Model)
# ---------------------------------------------------------------------------
# These are default values used when no per-vehicle override is provided.
# All values are in SI units.

#: Minimum bumper-to-bumper gap at standstill (m).
IDM_S0_M: float = 2.0

#: Desired time headway — how many seconds of gap the driver wants (s).
IDM_T_S: float = 1.2

#: Maximum acceleration in free-driving conditions (m/s²).
IDM_A_MAX_MPS2: float = 1.5

#: Comfortable braking deceleration (m/s²). Positive value = deceleration.
IDM_B_MPS2: float = 2.0

#: IDM acceleration exponent. 4 is the standard value.
IDM_DELTA: int = 4

#: Assumed vehicle body length used to convert center-to-center distance
#: to a bumper-to-bumper gap (m). Vehicles are modelled as points internally,
#: so this adds a realistic physical exclusion zone around each vehicle.
IDM_VEHICLE_LENGTH_M: float = 4.5
