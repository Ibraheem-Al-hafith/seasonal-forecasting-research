from __future__ import annotations

import platform
import sys
from importlib import metadata


def environment_info() -> dict:
    libs = {}
    for lib in ("numpy", "pandas", "scipy", "statsmodels", "pydantic", "forecasting-research"):
        try:
            libs[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            libs[lib] = "unknown"
    return {"python": sys.version.split()[0], "platform": platform.platform(), "libraries": libs}
