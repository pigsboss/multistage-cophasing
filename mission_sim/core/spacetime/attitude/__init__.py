"""Attitude models package."""
from .base import AttitudeModel
from .config import AttitudeConfig
from .models import AnalyticalEarthAttitude, SPICEAttitude
from .interpolated import InterpolatedAttitude

__all__ = [
    "AttitudeModel",
    "AttitudeConfig",
    "AnalyticalEarthAttitude",
    "SPICEAttitude",
    "InterpolatedAttitude",
    "create_attitude_model",
]


def create_attitude_model(config: AttitudeConfig) -> AttitudeModel:
    """Factory function that returns the appropriate AttitudeModel based on config.

    Priority:
        1. mode == "spice" or ("auto" and SPICE is available): SPICEAttitude
        2. mode == "analytical" (or fallback): AnalyticalEarthAttitude (only for Earth)
        3. mode == "interpolated": InterpolatedAttitude

    Args:
        config: AttitudeConfig instance.

    Returns:
        AttitudeModel instance.

    Raises:
        ValueError: If mode is unsupported or body has no analytical model.
    """
    mode = config.mode.lower()

    # ---- SPICE mode ----
    if mode in ("spice", "auto"):
        try:
            return SPICEAttitude(config)
        except (ImportError, RuntimeError):
            if mode == "spice":
                raise
            # fall through to analytical if auto

    # ---- Interpolated mode ----
    if mode == "interpolated":
        if config.times is None or config.quaternions is None:
            raise ValueError("Interpolated mode requires 'times' and 'quaternions' in config.")
        return InterpolatedAttitude(
            times=config.times,
            quaternions=config.quaternions,
            frame_from=config.frame_from,
            frame_to=config.frame_to,
        )

    # ---- Analytical mode ----
    if config.body.lower() == "earth":
        return AnalyticalEarthAttitude()
    else:
        raise ValueError(
            f"No analytical attitude model available for body '{config.body}'. "
            f"Use mode='spice' or mode='interpolated'."
        )
