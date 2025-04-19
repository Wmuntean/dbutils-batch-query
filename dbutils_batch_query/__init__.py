# ====================================================================
# Author: William Muntean
# Copyright (C) 2025 William Muntean. All rights reserved.
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://opensource.org/licenses/MIT
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ====================================================================

"""
=======================================================
dbutils_batch_query
=======================================================

This package provides a unified interface for batch querying, prompt management,
and environment configuration for Databricks utilities and LLM-powered workflows.

Project Overview
================

`dbutils_batch_query` is designed to streamline the process of running batch queries against Databricks
and managing prompt templates for LLM applications. It supports environment configuration via `.env` files,
modular prompt loading, and utilities for extracting and processing model outputs.

Key Features
============

1. **Batch Model Querying**:

   - Efficiently submit and manage batch queries to Databricks or LLM endpoints.
   - Includes error handling and default return structures for robust workflows.

2. **Prompt Management**:

   - Load, organize, and retrieve prompt templates for LLM applications.
   - Supports flexible prompt storage and retrieval patterns.

3. **Environment Configuration**:

   - Loads environment variables from a `.env` file in the current working directory.
   - Checks if running in a Databricks Notebook and loads credentials from workspace automatically.

.. Note::
    - The `.env` file should define ``DATABRICKS_TOKEN`` and ``DATABRICKS_HOST`` for Databricks API access.
    - Ensure the `.env` file is present in the directory where your script is executed unless set globally.

.. currentmodule:: dbutils_batch_query

Modules
=======

.. automodule:: dbutils_batch_query.model_query
    :no-members:
    :no-inherited-members:
    :no-special-members:

.. automodule:: dbutils_batch_query.prompts
    :no-members:
    :no-inherited-members:
    :no-special-members:
"""

__author__ = "William Muntean"
__email__ = "williamjmuntean@gmail.com"
__license__ = "MIT"
__maintainer__ = "William Muntean"
__date__ = "2025-04-18"

import importlib.metadata
import subprocess
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

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


# ==================================================================
# Load env secrets
# ==================================================================
load_dotenv(override=False)

from .model_query import (  # noqa: E402
    batch_model_query,
    extract_json_items,
    with_default_return,
)
from .prompts import load_all, load_prompt  # noqa: E402

__all__ = [
    "batch_model_query",
    "extract_json_items",
    "with_default_return",
    "load_all",
    "load_prompt",
]
