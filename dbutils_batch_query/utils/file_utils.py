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
Databricks File Utilities
=======================================================

This module provides utilities to download, upload, and delete files and
directories from Databricks volumes.

The file operations are performed in the following stages:

1. **Initialization**:
   - Instantiate ``WorkspaceClient`` and prepare paths.
2. **Operation**:
   - For download/upload/delete, handle single items or recurse through directories.
3. **Completion**:
   - Print progress messages and finalize the operation.

.. Note::
   - Environment variables ``DATABRICKS_TOKEN`` and ``DATABRICKS_HOST`` must be set.

.. Important::
   - Deletes are irreversible; use with caution.

.. currentmodule:: dbutils_batch_query.utils.file_utils

Functions
=========
.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: function_name_only.rst

   download_from_databricks
   upload_to_databricks
   delete_from_databricks

Standalone Execution
=====================
This module is not intended to be executed as a standalone script.
"""

__author__ = "William Muntean"
__email__ = "williamjmuntean@gmail.com"
__license__ = "MIT"
__maintainer__ = "William Muntean"
__date__ = "2025-05-09"

import shutil
from pathlib import Path

from databricks.sdk import WorkspaceClient


def download_from_databricks(
    remote_path: str,
    local_path: str | Path,
) -> None:
    """
    Download content from a Databricks volume to a local directory.

    This function can download either a single file or an entire folder (including
    all nested files and subdirectories) from a specified remote path to a local
    destination.

    Parameters
    ----------
    remote_path : str
        Path to the file or folder within the volume to download.
    local_path : str or Path
        Local directory path where files will be saved.

    Returns
    -------
    None
        This function does not return any value but prints progress information
        to standard output.

    .. Note::
        - Progress is printed to standard output during download.
        - Existing files at the destination will be overwritten without confirmation.
        - For single file downloads, the file will be saved in the local_path with its original name.

    .. Warning::
        - The function will fail if environment variables ``DATABRICKS_TOKEN`` and ``DATABRICKS_HOST`` are not set.

    Examples
    --------
    >>> # Download a directory
    >>> download_folder(
    ...     "data/reports",
    ...     "./local_data",
    ... )
    Downloaded: /path/to/local_data/file1.csv
    Downloaded: /path/to/local_data/subdir/file2.csv

    >>> # Download a single file
    >>> download_folder(
    ...     "data/reports/report.csv",
    ...     "./local_data",
    ... )
    Downloaded: /path/to/local_data/report.csv
    """
    # Construct absolute paths for source and destination
    remote_path = Path(remote_path)
    local_path = Path(local_path).resolve()

    # Ensure the local save path exists
    local_path.mkdir(parents=True, exist_ok=True)

    client = WorkspaceClient()

    # Check if remote_path is a file
    try:
        # Attempt to get file info - will fail if it's a directory
        file_info = client.files.get_metadata(remote_path.as_posix())

        local_file_path = local_path / remote_path.name
        download_response = client.files.download(remote_path.as_posix())
        with local_file_path.open("wb") as local_file:
            shutil.copyfileobj(download_response.contents, local_file)
        print(f"Downloaded: {local_file_path}")
        return
    except Exception as e:
        # If we get here and it's not because it's a directory, re-raise the exception
        if str(e) != "Not Found":
            raise

    def recursive_download(current_remote_path: Path, current_local_path: Path) -> None:
        # List contents of the current remote directory
        entries = client.files.list_directory_contents(current_remote_path.as_posix())

        for entry in entries:
            remote_entry_path = current_remote_path / entry.path
            local_entry_path = current_local_path / Path(entry.path).name

            if entry.is_directory:
                # Create local directory and process its contents
                local_entry_path.mkdir(parents=True, exist_ok=True)
                recursive_download(remote_entry_path, local_entry_path)
            else:
                # Download file content and save to local path
                download_response = client.files.download(remote_entry_path.as_posix())
                with local_entry_path.open("wb") as local_file:
                    shutil.copyfileobj(download_response.contents, local_file)
                print(f"Downloaded: {local_entry_path}")

    # Begin recursive download process
    recursive_download(remote_path, local_path)


def upload_to_databricks(
    remote_path: str,
    local_path: str | Path,
) -> None:
    """
    Upload a local file or directory to a Databricks volume.

    This function handles uploading either a single file or an entire directory
    structure to a Databricks volume. It preserves directory hierarchies when
    uploading folders.

    Parameters
    ----------
    remote_path : str
        Path within the volume where file(s) will be uploaded.
    local_path : str or Path
        Local file or directory path to upload.

    Returns
    -------
    None
        This function does not return any value but prints progress information
        to standard output.

    .. Note::
        - Progress is printed to standard output during upload.
        - Existing files at the destination will be overwritten without confirmation.
        - For single file uploads, the remote_path should include the target filename.
        - For directory uploads, the remote_path should be the target directory.

    .. Warning::
        - The function will fail if environment variables ``DATABRICKS_TOKEN`` and ``DATABRICKS_HOST`` are not set.

    Examples
    --------
    >>> # Upload a single file
    >>> upload_to_databricks(
    ...     "data/reports/report.csv",
    ...     "./local_data/report.csv",
    ... )
    Uploaded: ./local_data/report.csv to /Volumes/my_catalog/my_schema/my_volume/data/reports/report.csv

    >>> # Upload a directory
    >>> upload_to_databricks(
    ...     "data/reports",
    ...     "./local_data",
    ... )
    Uploaded: ./local_data/file1.csv to /Volumes/my_catalog/my_schema/my_volume/data/reports/file1.csv
    Uploaded: ./local_data/subdir/file2.csv to /Volumes/my_catalog/my_schema/my_volume/data/reports/subdir/file2.csv
    """
    # Convert paths to Path objects
    local_path = Path(local_path).resolve()
    remote_path = Path(remote_path)

    # Check if local source path exists
    if not local_path.exists():
        raise FileNotFoundError(f"Local source path does not exist: {local_path}")

    client = WorkspaceClient()

    # Handle single file upload
    if local_path.is_file():
        # Ensure directories exist
        try:
            client.files.get_directory_metadata(remote_path.as_posix())
        except Exception:
            # Directory doesn't exist, create it including all parents
            client.files.create_directory(remote_path.as_posix())

        remote_item_path = remote_path / local_path.name
        # Upload the file
        with local_path.open("rb") as local_file:
            client.files.upload(
                remote_item_path.as_posix(), contents=local_file, overwrite=True
            )
        print(f"Uploaded: {local_path} to {remote_path}")
        return

    # Handle directory upload with recursive function
    def recursive_upload(current_local_path: Path, current_remote_path: Path) -> None:
        # Ensure remote directory exists
        try:
            client.files.get_directory_metadata(current_remote_path.as_posix())
        except Exception:
            # Directory doesn't exist, create it
            client.files.create_directory(current_remote_path.as_posix())

        # Iterate through local directory contents
        for local_item in current_local_path.iterdir():
            remote_item_path = current_remote_path / local_item.name

            if local_item.is_dir():
                # Create remote directory and process its contents
                recursive_upload(local_item, remote_item_path)
            else:
                # Upload the file
                with local_item.open("rb") as local_file:
                    client.files.upload(
                        remote_item_path.as_posix(), contents=local_file, overwrite=True
                    )
                print(f"Uploaded: {local_item} to {remote_item_path}")

    # Begin recursive upload process
    recursive_upload(local_path, remote_path)


def delete_from_databricks(remote_path: str) -> None:
    """
    Delete a file or directory from a Databricks volume.

    Parameters
    ----------
    remote_path : str
        Path within the volume to delete. Can point to a single file or a directory.

    Returns
    -------
    None
        This function does not return any value but prints progress information
        to standard output.

    .. Warning::
        - Deletions are irreversible; ensure the correct path is specified.

    Examples
    --------
    >>> delete_from_databricks("data/reports/report.csv")
    Deleted: data/reports/report.csv

    >>> delete_from_databricks("data/old_reports")
    Deleted directory: data/old_reports
    """
    client = WorkspaceClient()
    # Try deleting as a file
    try:
        client.files.delete(remote_path)
        print(f"Deleted: {remote_path}")
        return
    except Exception as error_file:
        try:
            client.files.delete_directory(remote_path)
            print(f"Deleted directory: {remote_path}")
        except Exception as error_directory:
            raise error_directory
