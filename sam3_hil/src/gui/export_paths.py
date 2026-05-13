"""Export path helpers for STAMP GUI."""

import sys
from pathlib import Path
from typing import Optional


def get_default_export_dir(
    *,
    is_frozen: Optional[bool] = None,
    home: Optional[Path] = None,
) -> Path:
    """Return the default export directory for source and packaged runs."""
    if is_frozen is None:
        is_frozen = bool(getattr(sys, "frozen", False))

    if is_frozen:
        home_dir = Path(home) if home is not None else Path.home()
        return home_dir / "data" / "output"

    project_root = Path(__file__).resolve().parents[2]
    return project_root / "data" / "output"


def get_default_log_dir(
    *,
    is_frozen: Optional[bool] = None,
    home: Optional[Path] = None,
) -> Path:
    """Return the default action-log directory next to exported outputs."""
    return get_default_export_dir(is_frozen=is_frozen, home=home) / "logs"
