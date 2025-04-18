# Copilot Documentation Style Guide

## Header File Docstrings

Use Sphinx style `rst` syntax for header file docstrings. Include autosummary for better documentation generation with Sphinx. Use double backticks for inline code and Sphinx directives for notes and warnings. Here is a detailed template to follow:

```rst
"""
=======================================================
Module Name
=======================================================

This module provides functionality to [brief description of the module's purpose].

[Detailed section on any important algorithms or processes]
===========================================================

The [algorithm/process] is performed in the following stages:

1. **Initialization**:
   
   - Description of the initialization step.

2. [Step two in algorithm/process]:
   
   - Description of the second step.

3. [Step three in algorithm/process]:
   
   - Description of the next step.

.. Note::
    - Any important notes related to the algorithm/process.

.. Important::
    - Any critical information or warnings.

.. currentmodule:: [module_name]

Functions
=========

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: function_name_only.rst

    [function1]
    [function2]

Standalone Execution
=====================
When run as a standalone script, this module processes [description of the data/process] and outputs the results to [output format].

- Output Files:
    - ``output_file1``
    - ``output_file2``

.. Note::
   The following paths must be correctly set within the script's ``__main__`` block for successful
   execution:

   - ``dir_path``: Defines the [describe the purpose of the path].

.. code-block:: bash

    python [script_name].py

"""
```

## Python Function/Method/Class Docstrings

Use numpydoc style for Python function/method/class docstrings, incorporating Sphinx style `rst` syntax. Use double backticks for inline code. Here's a template:

```python
"""
Summary line.

Extended description of the function/method/class.

Parameters
----------
param1 : type
    Description of parameter ``param1``.
param2 : type, optional
    Description of parameter ``param2``. Default is ``default_value``.

Returns
-------
DataFrame or type
    Description of the returned object.

.. Note::
    Additional notes can be included here.

.. Warning::
    Any warnings related to the function/method/class.

Examples
--------
Examples should be written in doctest format, and should illustrate how to use the function/method/class.

>>> example_function(param1, param2)
result
"""

def example_function(param1, param2=None):
    # Function implementation
    pass
```

### Documenting pandas DataFrame Columns

When a function/method returns a pandas DataFrame, include a detailed description of the DataFrame's columns. Use double backticks for inline code and Sphinx directives for notes and warnings. Here's a template:

```python
"""
Returns
-------
pd.DataFrame
    [Short Dataframe Description]:

    - ``column 1``: [Short description of column].
    - ``column 2``: [Short description of column].
    - ``column 3``: [Short description of column].
"""
```

## Type Hints for Functions

For Python 3.12 and above, include type hints for functions to improve code readability and maintainability. Use type hints from the standard library whenever possible. Only import from the `typing` library when there are no other suitable approaches.

### Example

```python
def example_function(param1: int, param2: str = "default_value") -> bool:
    """
    Summary line.

    Extended description of the function.

    Parameters
    ----------
    param1 : int
        Description of parameter `param1`.
    param2 : str, optional
        Description of parameter `param2`. Default is `"default_value"`.

    Returns
    -------
    bool
        Description of the returned object.
    """
    # Function implementation
    return True
```

### Guidelines

- Always use type hints for function parameters and return types.
- Prefer built-in types (e.g., `int`, `str`, `list`, `dict`, `|`) over importing from `typing` when possible.
- Use `from __future__ import annotations` to avoid circular import issues and to forward declare types.
- Only import from the `typing` library when absolutely necessary.

## Additional Notes

- Ensure that docstrings are clear and concise.
- Use proper grammar and punctuation.
- Include all relevant information to make the documentation useful for users.
- Keep the documentation up-to-date with any changes in the code.
