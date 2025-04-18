"""
=======================================================
Data Import Module
=======================================================

This module provides functionality to import and preprocess data for analysis.

Data Import Process
==================

The data import process is performed in the following stages:

1. **Data Loading**:
   
   - Loading data from various sources (CSV, Excel, databases).
   - Handling different file formats and encodings.

2. **Data Cleaning**:
   
   - Handling missing values.
   - Removing duplicates and outliers.
   - Standardizing data formats.

3. **Data Transformation**:
   
   - Feature engineering and selection.
   - Normalization and scaling.
   - Encoding categorical variables.

.. Note::
    - Ensure all data sources are properly configured before import.
    - Large datasets may require chunked processing.

.. Important::
    - Always validate data integrity after import.

.. currentmodule:: dbutils_batch_query.data_import

Functions
=========

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: function_name_only.rst

    load_csv
    load_excel
    clean_data
    transform_data

"""
from pathlib import Path
from typing import Union, Optional, Dict, Any

import pandas as pd
import numpy as np


def load_csv(
    file_path: Union[str, Path], 
    date_cols: Optional[list[str]] = None,
    **kwargs: Any
) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Loads data from a CSV file into a pandas DataFrame with options
    for handling date columns and additional pandas read_csv parameters.

    Parameters
    ----------
    file_path : str or Path
        Path to the CSV file to be loaded.
    date_cols : list of str, optional
        List of column names to parse as dates. Default is ``None``.
    **kwargs : dict
        Additional keyword arguments to pass to ``pd.read_csv``.

    Returns
    -------
    pd.DataFrame
        The loaded data as a DataFrame:
        
        - Each column from the CSV file becomes a column in the DataFrame.
        - Column types are inferred or explicitly set through parameters.
        - Date columns are properly parsed if specified.

    Notes
    -----
    This function is a wrapper around pandas read_csv with improved date handling.

    Examples
    --------
    >>> df = load_csv("data.csv", date_cols=["date_column"])
    >>> df.dtypes
    column1       int64
    date_column   datetime64[ns]
    column3       object
    dtype: object
    """
    if date_cols is None:
        date_cols = []
    
    return pd.read_csv(
        file_path,
        parse_dates=date_cols,
        **kwargs
    )


def load_excel(
    file_path: Union[str, Path],
    sheet_name: Union[str, int] = 0,
    date_cols: Optional[list[str]] = None,
    **kwargs: Any
) -> pd.DataFrame:
    """
    Load data from an Excel file.

    Loads data from an Excel file into a pandas DataFrame with options
    for specifying the sheet and handling date columns.

    Parameters
    ----------
    file_path : str or Path
        Path to the Excel file to be loaded.
    sheet_name : str or int, optional
        Name or index of the sheet to load. Default is ``0`` (first sheet).
    date_cols : list of str, optional
        List of column names to parse as dates. Default is ``None``.
    **kwargs : dict
        Additional keyword arguments to pass to ``pd.read_excel``.

    Returns
    -------
    pd.DataFrame
        The loaded data as a DataFrame:
        
        - Each column from the Excel sheet becomes a column in the DataFrame.
        - Column types are inferred or explicitly set through parameters.
        - Date columns are properly parsed if specified.

    Examples
    --------
    >>> df = load_excel("data.xlsx", sheet_name="Sheet1", date_cols=["date_column"])
    >>> df.head()
       column1 date_column  column3
    0        1  2023-01-01     data1
    1        2  2023-01-02     data2
    2        3  2023-01-03     data3
    """
    if date_cols is None:
        date_cols = []
    
    df = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        **kwargs
    )
    
    # Convert date columns
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    return df


def clean_data(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    fill_na: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Clean a DataFrame by handling missing values and duplicates.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to clean.
    drop_duplicates : bool, optional
        Whether to drop duplicate rows. Default is ``True``.
    fill_na : dict, optional
        Dictionary mapping column names to values used to fill NaN values.
        Default is ``None``.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame:
        
        - Duplicate rows removed (if requested)
        - Missing values filled (if specified)
        - Original column order preserved

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'col1': [1, 2, 2, None],
    ...     'col2': ['a', 'b', 'b', 'd']
    ... })
    >>> clean_df = clean_data(df, fill_na={'col1': 0})
    >>> clean_df
       col1 col2
    0     1    a
    1     2    b
    3     0    d
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Drop duplicates if requested
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    
    # Fill NaN values if specified
    if fill_na:
        for col, value in fill_na.items():
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].fillna(value)
    
    return cleaned_df


def transform_data(
    df: pd.DataFrame,
    normalize_cols: Optional[list[str]] = None,
    log_transform_cols: Optional[list[str]] = None,
    categorical_cols: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Transform data by normalizing, log-transforming, and encoding categorical variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to transform.
    normalize_cols : list of str, optional
        List of columns to normalize (min-max scaling). Default is ``None``.
    log_transform_cols : list of str, optional
        List of columns to log transform. Default is ``None``.
    categorical_cols : list of str, optional
        List of categorical columns to one-hot encode. Default is ``None``.

    Returns
    -------
    pd.DataFrame
        Transformed DataFrame:
        
        - Normalized columns (scaled between 0 and 1)
        - Log-transformed columns (natural logarithm)
        - One-hot encoded categorical columns

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'value': [1, 10, 100, 1000],
    ...     'category': ['A', 'B', 'A', 'C']
    ... })
    >>> transform_data(df, 
    ...                normalize_cols=['value'], 
    ...                categorical_cols=['category'])
       value  category_A  category_B  category_C
    0   0.000        1.0        0.0        0.0
    1   0.009        0.0        1.0        0.0
    2   0.099        1.0        0.0        0.0
    3   1.000        0.0        0.0        1.0
    """
    # Create a copy to avoid modifying the original
    transformed_df = df.copy()
    
    # Normalize columns
    if normalize_cols:
        for col in normalize_cols:
            if col in transformed_df.columns:
                min_val = transformed_df[col].min()
                max_val = transformed_df[col].max()
                if max_val > min_val:
                    transformed_df[col] = (transformed_df[col] - min_val) / (max_val - min_val)
    
    # Log transform columns
    if log_transform_cols:
        for col in log_transform_cols:
            if col in transformed_df.columns:
                # Add a small constant to avoid log(0)
                transformed_df[col] = np.log1p(transformed_df[col])
    
    # One-hot encode categorical columns
    if categorical_cols:
        for col in categorical_cols:
            if col in transformed_df.columns:
                dummies = pd.get_dummies(transformed_df[col], prefix=col)
                transformed_df = pd.concat([transformed_df, dummies], axis=1)
                transformed_df = transformed_df.drop(col, axis=1)
    
    return transformed_df


if __name__ == "__main__":
    # Example usage of the module
    sample_data = {
        'value': [1, 10, 100, 1000, None],
        'category': ['A', 'B', 'A', 'C', 'B'],
        'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original data:")
    print(df)
    
    # Clean the data
    cleaned_df = clean_data(df, fill_na={'value': 0})
    print("\nCleaned data:")
    print(cleaned_df)
    
    # Transform the data
    transformed_df = transform_data(
        cleaned_df,
        normalize_cols=['value'],
        categorical_cols=['category']
    )
    print("\nTransformed data:")
    print(transformed_df)