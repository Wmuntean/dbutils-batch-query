"""
=======================================================
Command Line Interface
=======================================================

This module provides a command-line interface for the dbutils_batch_query package.

.. currentmodule:: dbutils_batch_query.cli

Functions
=========

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: function_name_only.rst

    main

"""

import click
from . import __version__


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__)
def main():
    """Batch query databricks foundation LLM models"""
    pass


@main.command()
@click.argument("name", default="World")
def hello(name):
    """Say hello to NAME (default: "World")"""
    click.echo(f"Hello, {name}!")


if __name__ == "__main__":
    main()