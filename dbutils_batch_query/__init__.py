import importlib.metadata
import subprocess
from pathlib import Path
from typing import Any

__package_version = "unknown"


def __get_package_version() -> str:
    global __package_version
    if __package_version != "unknown":
        return __package_version

    root = Path(__file__).resolve().parents[1]
    is_git_repo = (root / ".git").exists()

    if is_git_repo:
        try:
            version = (
                subprocess.check_output(
                    ["git", "describe", "--tags", "--dirty", "--always"],
                    cwd=root,
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
            __package_version = f"{version}+"
            return __package_version
        except Exception:
            pass  # fallback below

    # Not a git repo or git failed — use installed version
    try:
        __package_version = importlib.metadata.version("dbutils_batch_query")
    except importlib.metadata.PackageNotFoundError:
        __package_version = "unknown"

    return __package_version


def __getattr__(name: str) -> Any:
    if name in ("version", "__version__"):
        return __get_package_version()
    raise AttributeError(f"No attribute {name} in module {__name__}.")
