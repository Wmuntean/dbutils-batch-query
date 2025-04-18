import argparse
import re
import subprocess
import sys
from pathlib import Path

import tomlkit

python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
python_constraint = f">={python_version},<4.0"

parser = argparse.ArgumentParser(
    description="Generate environment.yml and update pyproject.toml"
)
parser.add_argument(
    "--use-pigar",
    "-pi",
    action="store_true",
    help="Run pigar to create the requirements.txt file",
)
parser.add_argument(
    "--use-poetry", "-po", action="store_true", help="Add dependencies with Poetry"
)
args, unknown = parser.parse_known_args()

# Prompt if no args
if not (args.use_pigar or args.use_poetry):
    print("=== Requirements File Generator ===")
    args.use_pigar = (
        input("Run pigar to create requirements.txt? (y/n): ")
        .strip()
        .lower()
        .startswith("y")
    )
    args.use_poetry = (
        input("Add dependencies with Poetry? (y/n): ").strip().lower().startswith("y")
    )

# Project root
project_root = Path.cwd().parent.resolve()
# project_root = Path.cwd().resolve()
print(f"Using project root: {project_root}")
req_file = project_root / "requirements.txt"
env_file = project_root / "environment.yml"
poetry_file = project_root / "pyproject.toml"

# Get project name from pyproject
with open(poetry_file, "r", encoding="utf-8") as f:
    toml_data = tomlkit.load(f)  # use tomlkit to load

project_name = toml_data.get("project", {}).get("name") or toml_data.get(
    "tool", {}
).get("poetry", {}).get("name", "your-project-name")

# Run pigar if requested
if args.use_pigar:
    subprocess.run(["pigar", "gen", "-f", str(req_file)], check=True, cwd=project_root)
    input(
        "Press Enter to continue after reviewing `requirements.txt`, or Ctrl+C to abort."
    )

# Parse requirements.txt
if not req_file.exists():
    raise FileNotFoundError(f"`requirements.txt` not found at {req_file}")

with open(req_file, "r") as f:
    requirements = f.readlines()

pip_dependencies = []
for line in requirements:
    match = re.match(r"([\w\-.]+)([=<>!~]*[\d\w\.\*]*)", line.strip())
    if match:
        package, version = match.groups()
        match = re.search(r"[=<>!~]+([\w\.\-]+)", version)
        if match:
            version = match.group(1)
            dep = f"{package}~={version}"
        else:
            version = "0.0.0"  # Default version if none is found
            dep = f"{package}>{version}"
            print(
                f"Warning: No version found for package '{package}'. Using default version '{version}'."
            )
        pip_dependencies.append(dep)

# Write environment.yml
yml_content = [
    f"name: {project_name}",
    "channels:",
    "  - conda-forge",
    "  - defaults",
    "dependencies:",
    f"  - python={python_version}.*",
    "  - pip",
    "  - pip:",
]
yml_content.extend(f"    - {dep}" for dep in pip_dependencies)
with open(env_file, "w") as f:
    f.write("\n".join(yml_content))
print(f"`environment.yml` written to {env_file}")

# --- Update pyproject.toml using tomlkit ---
if "project" not in toml_data:
    toml_data["project"] = {}

toml_data["project"]["requires-python"] = python_constraint
toml_data["project"]["dependencies"] = sorted(pip_dependencies)

# Optional: if using poetry, add via CLI too
if args.use_poetry:
    print("Running poetry to add dependencies...")
    subprocess.run(["poetry", "add"] + pip_dependencies, check=True, cwd=project_root)

    try:
        result = subprocess.run(
            ["poetry", "self", "show", "plugins"],
            check=True,
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        if "poetry-plugin-export" not in result.stdout:
            subprocess.run(
                ["poetry", "self", "add", "poetry-plugin-export"],
                check=True,
                cwd=project_root,
            )
    except subprocess.CalledProcessError as e:
        print("Could not verify or install poetry-plugin-export:", e)

    subprocess.run(
        [
            "poetry",
            "export",
            "-f",
            "requirements.txt",
            "--without-hashes",
            "-o",
            "requirements-exact.txt",
        ],
        check=True,
        cwd=project_root,
    )
    print("Exact requirements exported to requirements-exact.txt")

# Write final toml
with open(poetry_file, "w", encoding="utf-8") as f:
    f.write(tomlkit.dumps(toml_data))  # use tomlkit to dump
print(f"`pyproject.toml` updated at {poetry_file}")
