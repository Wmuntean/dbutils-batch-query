# dbutils-batch-query

Batch query databricks foundation LLM models

## Project Overview

[Insert project overview]

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


- API documentation is available at https://Wmuntean.github.io/dbutils_batch_query/




## Tests

To run the tests:

```bash
pytest
```


## Usage

```python
import dbutils_batch_query

# Basic usage example
print(f"{{ project_name }} version: {{{ project_name }}.__version__}")
```

<!-- data_import -->
### Data Import Module

The `dbutils_batch_query.data_import` module provides utilities for loading and preprocessing data from various sources.

```python
from dbutils_batch_query.data_import import load_csv, clean_data, transform_data

# Load data from a CSV file
df = load_csv("path/to/data.csv", date_cols=["date_column"])

# Clean the data
cleaned_df = clean_data(df, drop_duplicates=True, fill_na={"numeric_column": 0})

# Transform the data
transformed_df = transform_data(
    cleaned_df,
    normalize_cols=["numeric_feature"],
    categorical_cols=["category_column"]
)
```

Key features:
- Load data from CSV and Excel files with automatic date parsing
- Clean datasets by handling missing values and duplicates
- Transform data with normalization, log transformations, and categorical encoding
<!-- module_name 1 -->

<!-- module_name 2 -->
### Statistical Analysis

The `dbutils_batch_query.example_analysis` module provides statistical analysis functions.

```python
import pandas as pd
from dbutils_batch_query.example_analysis import descriptive_stats, correlation_analysis, time_series_analysis

# Create or load your DataFrame
df = pd.DataFrame({...})

# Get descriptive statistics
stats_df, plots = descriptive_stats(df)
print(stats_df)

# Perform correlation analysis
corr_df, heatmap = correlation_analysis(df, method='pearson', threshold=0.3)
print(corr_df)

# Time series analysis for data with a datetime index
ts_data = pd.Series([...], index=pd.date_range(...))
results, plots = time_series_analysis(ts_data, seasonal=True)
```

Key capabilities:
- Compute descriptive statistics with distribution visualization
- Analyze correlations between variables with customizable heatmaps
- Run statistical tests including t-tests with visual interpretation
- Perform time series decomposition and basic forecasting
<!-- end modules -->



## Contributors
- [@Wmuntean](https://github.com/Wmuntean)

## License and IP Notice


This project is licensed under the MIT License - see the LICENSE file for details.

***