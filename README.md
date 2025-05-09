# dbutils-batch-query

![Python Versions](https://img.shields.io/badge/python-3.11%20|%203.12-blue)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![Build Status](https://github.com/Wmuntean/dbutils-batch-query/actions/workflows/python-test.yml/badge.svg)


Batch query databricks foundation LLM models asynchronously.

## Project Overview

This project provides utilities to interact with Databricks foundation language models efficiently. It allows for batch querying of models, handling prompt templating using Jinja2, processing model responses, and managing results with robust error handling and metadata tracking. Key features include asynchronous API calls for improved throughput, rate limiting, and saving intermediate/final results.

## Installation

### Option 1: Install package

Install specific release tag:
```bash
pip install -U git+https://github.com/Wmuntean/dbutils-batch-query.git@v0.1.0
```
To update to the latest unreleased version, you must remove previously installed version first:
```bash
pip uninstall dbutils_batch_query
pip install -U git+https://github.com/Wmuntean/dbutils-batch-query.git
```

### Option 2: Clone repository:

Clone the repository:
```bash
git clone https://github.com/Wmuntean/dbutils-batch-query.git
cd dbutils-batch-query
```

(Optional) Install requirements in working environment:
```bash
pip install -r requirements.txt
```
or
```bash
poetry install
```

Set system path in python editor:
```python
import sys
sys.path.insert(0, R"path\to\dbutils_batch_query")

# Confirm development version
import dbutils_batch_query
print(dbutils_batch_query.__version__)
# Should output: {version}+
```

## Documentation

- API documentation is available at [Documentation](https://wmuntean.github.io/dbutils-batch-query/).

## Usage

This package provides functions for querying Databricks models in batches and managing prompt templates.
<!-- batch_model_query -->
### Batch Model Querying
<!-- start Example -->
The [`batch_model_query`](#dbutils_batch_query.model_query.batch_model_query) function allows you to send multiple prompts to a specified model asynchronously.

#### Running in a Notebook
```python
from dbutils_batch_query.model_query import batch_model_query, extract_json_items
from dbutils_batch_query.prompts import load_prompt

user_prompt = load_prompt(
    "path/to/your/prompt_template.md", input_text="Some text to analyze."
)
# Example prompt information (replace with your actual prompts)
prompt_info = [
    {
        "system": "You are an assistant that extracts key information. Respond in a JSON codeblock.",
        "user": user_prompt,
        "id": "query_1",  # Optional: Add identifiers or other metadata
    },
    {
        "system": "You are an assistant that summarizes text. Respond in a JSON codeblock.",
        "user": load_prompt(
            "path/to/your/summary_template.md",
            document="Another document to summarize.",
        ),
        "id": "query_2",
    },
]

results = await batch_model_query(
    prompt_info=prompt_info,
    model="databricks-llama-4-maverick",  # Specify your Databricks model endpoint
    process_func=extract_json_items,  # Optional: function to process raw text response
    batch_size=5, # Optional: Batch size before optional save
    max_concurrent_requests=3, # Optional: Max concurrent requests
    rate_limit=(2, 1), # Optional: Number of requests per second
    results_path="output_results/",  # Optional: path to save results
    run_name="my_batch_run",  # Optional: identifier for the run
    # token and host are automatically fetched from environment or dbutils if not provided
)

# Process results
for result in results:
    if result["error"]:
        print(f"Error processing prompt {result.get('id', 'N/A')}: {result['error']}")
    else:
        print(f"Result for prompt {result.get('id', 'N/A')}:")
        # Access raw message or processed response
        # print(result["message"])
        print(result["processed_response"])

```

#### Running in a Python File
```python
import asyncio
from dbutils_batch_query.model_query import (
    batch_model_query,
    extract_json_items
)
from dbutils_batch_query.prompts import load_prompt

user_prompt = load_prompt(
    "path/to/your/prompt_template.md", input_text="Some text to analyze."
)
# Example prompt information (replace with your actual prompts)
prompt_info = [
    {
        "system": "You are an assistant that extracts key information. Respond in a JSON codeblock.",
        "user": user_prompt,
        "id": "query_1",  # Optional: Add identifiers or other metadata
    },
    {
        "system": "You are an assistant that summarizes text. Respond in a JSON codeblock.",
        "user": load_prompt(
            "path/to/your/summary_template.md",
            document="Another document to summarize.",
        ),
        "id": "query_2",
    },
]

results = asyncio.run(
    batch_model_query(
        prompt_info=prompt_info,
        model="databricks-llama-4-maverick",  # Specify your Databricks model endpoint
        process_func=extract_json_items,  # Optional: function to process raw text response
        batch_size=5, # Optional: Batch size before optional save
        max_concurrent_requests=3, # Optional: Max concurrent requests
        rate_limit=(2, 1), # Optional: Number of requests per second
        results_path="output_results/",  # Optional: path to save results
        run_name="my_batch_run",  # Optional: identifier for the run
        # token and host are automatically fetched from environment or dbutils if not provided
    )
)

# Process results
for result in results:
    if result["error"]:
        print(f"Error processing prompt {result.get('id', 'N/A')}: {result['error']}")
    else:
        print(f"Result for prompt {result.get('id', 'N/A')}:")
        # Access raw message or processed response
        # print(result["message"])
        print(result["processed_response"])
```
<!-- end Example -->

Key features:
- Asynchronous processing of multiple prompts.
- Configurable batch size and concurrency limits.
- Optional response processing function.
- Automatic handling of Databricks token and host (via environment variables or `dbutils` in Databricks Notebook).
- Saves intermediate and final results (pickle and parquet formats) if `results_path` and `run_name` are provided.
- Detailed metadata included in results (timing, token usage, errors).

### Prompt Template Management

The [`prompts`](#dbutils_batch_query.prompts) module helps manage Jinja2 prompt templates.

**Load all templates from a directory:**

```python
from dbutils_batch_query.prompts import load_all

# Load all .md templates from the specified directory
prompt_templates = load_all("path/to/prompt_templates/")

# Access a specific template
template = prompt_templates["my_template_name"] # Key is the filename stem

# Render the template
# rendered = template.render(variable="value")
```

**Load and render a single template:**

```python
from dbutils_batch_query.prompts import load_prompt

# Load and render a template file with arguments
rendered_prompt = load_prompt(
    "path/to/prompt_templates/my_template_name.md",
    input_text="Some dynamic content for the prompt."
)

print(rendered_prompt)
```

### File Management

The [`file_utils`](#dbutils_batch_query.utils.file_utils) module provides utilities to manage files and directories on Databricks volumes. It supports downloading, uploading, and deleting both individual files and entire folder trees.

**Functions:**
- ``download_from_databricks(remote_path: str, local_path: str | Path)``  
  Download a file or directory from a Databricks volume to a local path.
- ``upload_to_databricks(remote_path: str, local_path: str | Path)``  
  Upload a local file or directory to a Databricks volume. Directory hierarchies are preserved.
- ``delete_from_databricks(remote_path: str)``  
  Delete a file or directory (recursively) from a Databricks volume.

**Usage examples:**

```python
from dbutils_batch_query import (
    download_from_databricks,
    upload_to_databricks,
    delete_from_databricks,
)

# Download an entire folder
download_from_databricks("data/reports", "./local_reports")

# Download a single file
download_from_databricks("data/reports/report.csv", "./local_reports")

# Upload a local directory
upload_to_databricks("data/processed", "./processed_data")

# Upload a single file
upload_to_databricks("data/processed/summary.json", "./processed_data/summary.json")

# Delete a file or folder
delete_from_databricks("data/old_reports")
```

## Contributors
- [@Wmuntean](https://github.com/Wmuntean)

## License and IP Notice


This project is licensed under the MIT License - see the LICENSE file for details.

***