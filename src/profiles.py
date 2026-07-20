"""
DriveSense — per-driver calibration profiles.

Profiles are stored as JSON files under data/profiles/{driver_name}.json.
Each file holds the calibrated EAR threshold so it persists across restarts.

Usage
-----
    import profiles
    thresh = profiles.load_threshold("Davis")   # returns None if no profile yet
    profiles.save_threshold("Davis", 0.34)
    all_names = profiles.list_drivers()
"""
import json
import os
import re

_ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROFILES_DIR = os.path.join(_ROOT, "data", "profiles")


def _profile_path(name: str) -> str:
    # Sanitise to alphanumerics + underscore/hyphen to avoid path traversal
    safe = re.sub(r"[^\w\- ]", "", name).strip().replace(" ", "_")
    if not safe:
        raise ValueError(f"Invalid driver name: {name!r}")
    return os.path.join(_PROFILES_DIR, f"{safe}.json")


def load_threshold(name: str) -> float | None:
    """Return the saved EAR threshold for *name*, or None if not found."""
    try:
        with open(_profile_path(name)) as f:
            data = json.load(f)
        return float(data["ear_threshold"])
    except (FileNotFoundError, KeyError, ValueError):
        return None


def save_threshold(name: str, ear_threshold: float) -> None:
    """Persist the calibrated EAR threshold for *name*."""
    os.makedirs(_PROFILES_DIR, exist_ok=True)
    path = _profile_path(name)
    # Preserve any existing keys, just update ear_threshold
    try:
        with open(path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    data["ear_threshold"] = round(ear_threshold, 4)
    data["name"] = name
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def list_drivers() -> list[str]:
    """Return a sorted list of driver names that have saved profiles."""
    os.makedirs(_PROFILES_DIR, exist_ok=True)
    names = []
    for fname in os.listdir(_PROFILES_DIR):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(_PROFILES_DIR, fname)) as f:
                    data = json.load(f)
                names.append(data.get("name", fname[:-5]))
            except (json.JSONDecodeError, IOError):
                pass
    return sorted(names)
