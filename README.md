# dbutils-batch-query

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


- API documentation is available at [Wmuntean GitHub Pages](https://wmuntean.github.io/dbutils-batch-query/).

## Usage

This package provides functions for querying Databricks models in batches and managing prompt templates.
<!-- batch_model_query -->
### Batch Model Querying
<!-- start Example -->
The [`batch_model_query`](#dbutils_batch_query.model_query.batch_model_query) function allows you to send multiple prompts to a specified model asynchronously.

#### Running in a Notebook
```python
from dbutils_batch_query.model_query import (
    batch_model_query,
    extract_json_items,
    with_default_return,
)
from dbutils_batch_query.prompts import load_prompt

# Example prompt information (replace with your actual prompts)
prompt_info = [
    {
        "system": "You are an assistant that extracts key information.",
        "user": load_prompt(
            "path/to/your/prompt_template.md", input_text="Some text to analyze."
        ),
        "id": "query_1",  # Optional: Add identifiers or other metadata
    },
    {
        "system": "You are an assistant that summarizes text.",
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
    batch_size=5,
    max_concurrent_requests=3,
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
    extract_json_items,
    with_default_return,
)
from dbutils_batch_query.prompts import load_prompt

# Example prompt information (replace with your actual prompts)
prompt_info = [
    {
        "system": "You are an assistant that extracts key information.",
        "user": load_prompt(
            "path/to/your/prompt_template.md", input_text="Some text to analyze."
        ),
        "id": "query_1",  # Optional: Add identifiers or other metadata
    },
    {
        "system": "You are an assistant that summarizes text.",
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
        batch_size=5,
        max_concurrent_requests=3,
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

# Render the template (if needed, though batch_model_query handles rendering internally via load_prompt)
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

## Contributors
- [@Wmuntean](https://github.com/Wmuntean)

## License and IP Notice


This project is licensed under the MIT License - see the LICENSE file for details.

***