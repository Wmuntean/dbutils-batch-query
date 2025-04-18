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
prompts
=======================================================

This module provides utilities for loading and rendering Jinja2-based markdown prompt templates from directories or files.

Prompt Template Loading and Rendering
=====================================

The module supports recursive loading of all markdown templates in a directory and rendering individual templates with custom arguments.

.. Note::
    - Templates must use the ``.md`` extension and be compatible with Jinja2 syntax.

.. currentmodule:: dbutils_batch_query.prompts

Functions
=========

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: function_name_only.rst

    load_all
    load_prompt
"""

__author__ = "William Muntean"
__email__ = "williamjmuntean@gmail.com"
__license__ = "GPL v3"
__maintainer__ = "William Muntean"
__date__ = "2025-03-23"

from pathlib import Path

from jinja2 import Template


def load_all(directory_path: str | Path) -> dict[str, Template]:
    """
    Recursively load all markdown prompt templates from a directory into a dictionary.

    This function traverses the given directory and its subdirectories to find
    markdown template files (with .md extension) and loads them into Jinja2 Template objects.
    The templates are stored in a dictionary with filenames as keys, with folder
    names prepended in case of duplicate filenames.

    Parameters
    ----------
    directory_path : str or Path
        Path to the directory containing markdown prompt template files.

    Returns
    -------
    dict[str, Template]
        A dictionary where:

        - ``key``: Filename (with folder name prepended if duplicate).
        - ``value``: jinja Template object.

    .. Note::
        Template files are expected to be markdown files (.md extension) that can be parsed
        by Jinja2.

    Examples
    --------
    >>> templates = load_all("/path/to/templates")
    >>> admission_template = templates["admission_note"]
    """
    templates = {}
    directory = Path(directory_path)

    for file_path in directory.glob("**/*.md"):
        # Skip directories and hidden files
        if file_path.is_dir() or file_path.name.startswith("."):
            continue

        # Load the file content and create a template
        template_content = file_path.read_text()
        template = Template(template_content)

        # Use filename as key, handle duplicates by prepending folder name
        key = file_path.stem
        if key in templates:
            # Get the parent folder name
            folder_name = file_path.parent.name
            key = f"{folder_name}-{file_path.stem}"

        templates[key] = template

    return templates


def load_prompt(file_path: str | Path, **kwargs: list | str) -> str:
    """
    Load and render a single prompt template from a file.

    Parameters
    ----------
    file_path : str or Path
        Path to the template file to load.
    **kwargs : list or str
        Additional keyword arguments to pass to the template renderer.
        For example, you can pass ``examples`` as a list or string to provide
        example data for rendering.

    Returns
    -------
    str
        The rendered template content.

    Examples
    --------
    >>> prompt = load_prompt("/path/to/template.md")
    >>> prompt_with_examples = load_prompt(
    ...     "/path/to/template.md", examples=[]"example1", "example2"]
    ... )
    """
    template_path = Path(file_path)
    template_content = template_path.read_text()

    # Render the template with the provided keyword arguments
    template = Template(template_content)
    rendered_prompt = template.render(**kwargs)

    return rendered_prompt
