API Reference
=============

Documentation for all the main modules and functions of the dbutils_batch_query package.

The ``dbutils_batch_query`` package provides a unified interface for batch processing of model queries and prompt management. This documentation offers an overview of the core components of the API and provides links to detailed module documentation and key functions.

Modules Overview
----------------

* **Model Query**:  
  The :py:mod:`~dbutils_batch_query.model_query` module contains functions to handle model queries in batch. Key functions include:

  - :py:func:`~dbutils_batch_query.model_query.with_default_return`  
    Handles setting or retrieving default return values.
  - :py:func:`~dbutils_batch_query.model_query.extract_json_items`  
    Extracts JSON items from query responses.
  - :py:func:`~dbutils_batch_query.model_query.batch_model_query`  
    Executes and manages batch queries.


* **Prompts**:  
  The :py:mod:`~dbutils_batch_query.prompts` module manages prompt rendering and executions. For example:

  - :py:func:`~dbutils_batch_query.prompts.load_prompt`  
    Loads a single prompt with jinja rendering.

* **File Utilities**:
  The :py:mod:`~dbutils_batch_query.utils.file_utils` module provides utilities to interact with Databricks file systems. Key functions include:

  - :py:func:`~dbutils_batch_query.utils.file_utils.download_from_databricks`
    Downloads files or directories from Databricks volumes.
  - :py:func:`~dbutils_batch_query.utils.file_utils.upload_to_databricks`
    Uploads files or directories to Databricks volumes.
  - :py:func:`~dbutils_batch_query.utils.file_utils.delete_from_databricks`
    Deletes files or directories from Databricks volumes.

For more detailed insights into each module, please see:
   - :doc:`model_query`
   - :doc:`prompts`
   - :doc:`file_utils`
Quick Start Examples
--------------------
   .. include:: ../../../README.md
      :parser: myst_parser.sphinx_
      :start-after: <!-- start Example -->
      :end-before: <!-- end Example -->


.. Note::
   Ensure that your environment is correctly configured before executing batch queries. Refer to each module's documentation for detailed parameter descriptions and further examples.

.. toctree::
   :maxdepth: 2
   :hidden:
   :titlesonly:

   model_query
   prompts
   file_utils