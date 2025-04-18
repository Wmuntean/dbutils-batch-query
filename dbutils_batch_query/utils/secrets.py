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

""" """

__author__ = "William Muntean"
__email__ = "williamjmuntean@gmail.com"
__license__ = "MIT"
__maintainer__ = "William Muntean"
__date__ = "2025-04-18"

import os


def get_databricks_secrets() -> tuple[str, str]:
    """
    Retrieve Databricks secrets from environment variables or Databricks notebook context.

    This function first checks if it is running inside a Databricks notebook by looking for
    the ``DATABRICKS_RUNTIME_VERSION`` environment variable. If so, it attempts to retrieve
    the API token and host using the ``dbutils`` notebook context. Otherwise, it falls back
    to environment variables.

    Returns
    -------
    tuple[str, str]
        Tuple containing ``DATABRICKS_TOKEN`` and ``DATABRICKS_HOST``.

    Raises
    ------
    RuntimeError
        If either secret is missing.

    Examples
    --------
    >>> token, host = get_databricks_secrets()
    """
    token = None
    host = None

    if "DATABRICKS_RUNTIME_VERSION" in os.environ:
        try:
            from IPython import get_ipython

            dbutils = get_ipython().user_ns["dbutils"]
            token = (
                dbutils.notebook.entry_point.getDbutils()
                .notebook()
                .getContext()
                .apiToken()
                .get()
            )
            host = (
                dbutils.notebook.entry_point.getDbutils()
                .notebook()
                .getContext()
                .apiUrl()
                .get()
            )
        except Exception:
            token = None
            host = None

    if not token or not host:
        token = os.environ.get("DATABRICKS_TOKEN")
        host = os.environ.get("DATABRICKS_HOST")

    if not token or not host:
        raise RuntimeError(
            "DATABRICKS_TOKEN and DATABRICKS_HOST must be set in the environment or available via dbutils."
        )
    return token, host
