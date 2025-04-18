"""
=======================================================
Example Analysis Module
=======================================================

This module provides functionality for analyzing data with standard statistical
and visualization techniques.

Analysis Process
===========================================================

The analysis process is performed in the following stages:

1. **Data Exploration**:
   
   - Computing descriptive statistics.
   - Identifying patterns and relationships.
   - Visualizing distributions and correlations.

2. **Statistical Testing**:
   
   - Hypothesis testing.
   - Analysis of variance.
   - Correlation and regression analysis.

3. **Results Interpretation**:
   
   - Summarizing findings.
   - Generating insights.
   - Preparing visualizations.

.. Note::
    - Analysis functions handle pandas DataFrames as the primary data structure.
    - Most functions return both results and visualizations.

.. Important::
    - Ensure data is properly cleaned and preprocessed before analysis.
    - Validate statistical assumptions before interpreting results.

.. currentmodule:: dbutils_batch_query.example_analysis

Functions
=========

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: function_name_only.rst

    descriptive_stats
    correlation_analysis
    run_t_test
    time_series_analysis

"""
from typing import Dict, Tuple, List, Optional, Union
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns


def descriptive_stats(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    include_plots: bool = True,
    figsize: Tuple[int, int] = (12, 8)
) -> Tuple[pd.DataFrame, Optional[Dict[str, plt.Figure]]]:
    """
    Calculate descriptive statistics for numeric columns in a DataFrame.

    Computes common descriptive statistics and generates distribution plots
    for the specified numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the data to analyze.
    numeric_cols : list of str, optional
        List of numeric column names to analyze. If ``None``, all numeric columns are used.
        Default is ``None``.
    include_plots : bool, optional
        Whether to generate distribution plots. Default is ``True``.
    figsize : tuple of int, optional
        Figure size for plots. Default is ``(12, 8)``.

    Returns
    -------
    pd.DataFrame, dict or None
        Tuple containing:
        
        - DataFrame with descriptive statistics:
            - ``count``: Number of non-null observations
            - ``mean``: Mean of the observations
            - ``std``: Standard deviation
            - ``min``: Minimum value
            - ``25%``: First quartile
            - ``50%``: Median
            - ``75%``: Third quartile
            - ``max``: Maximum value
            - ``skew``: Skewness
            - ``kurtosis``: Kurtosis
        
        - Dictionary mapping column names to matplotlib Figure objects,
          or ``None`` if ``include_plots=False``

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'A': np.random.normal(0, 1, 100),
    ...     'B': np.random.exponential(1, 100),
    ...     'C': ['x', 'y'] * 50
    ... })
    >>> stats_df, plots = descriptive_stats(df)
    >>> stats_df
                   A          B
    count  100.000000 100.000000
    mean     0.017151   1.007473
    std      1.091267   0.964327
    min     -2.365464   0.005517
    25%     -0.678891   0.303829
    50%     -0.006345   0.719873
    75%      0.716429   1.385070
    max      2.959042   4.734513
    skew     0.218199   1.552458
    kurtosis -0.244298   2.763430
    """
    # Select numeric columns
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    else:
        # Ensure all specified columns exist and are numeric
        for col in numeric_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")
            if not np.issubdtype(df[col].dtype, np.number):
                raise ValueError(f"Column '{col}' is not numeric")
    
    # Calculate descriptive statistics
    stats_df = df[numeric_cols].describe()
    
    # Add skewness and kurtosis
    stats_df.loc['skew'] = df[numeric_cols].skew()
    stats_df.loc['kurtosis'] = df[numeric_cols].kurtosis()
    
    # Generate plots if requested
    plots = None
    if include_plots:
        plots = {}
        for col in numeric_cols:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            
            # Histogram with KDE
            sns.histplot(df[col], kde=True, ax=axes[0])
            axes[0].set_title(f'Distribution of {col}')
            axes[0].grid(True, linestyle='--', alpha=0.7)
            
            # Box plot
            sns.boxplot(y=df[col], ax=axes[1])
            axes[1].set_title(f'Box Plot of {col}')
            axes[1].grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plots[col] = fig
    
    return stats_df, plots


def correlation_analysis(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    method: str = 'pearson',
    threshold: float = 0.0,
    include_plot: bool = True,
    figsize: Tuple[int, int] = (10, 8)
) -> Tuple[pd.DataFrame, Optional[plt.Figure]]:
    """
    Perform correlation analysis on numeric columns in a DataFrame.

    Calculates correlation coefficients between numeric columns and optionally
    generates a correlation heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the data to analyze.
    numeric_cols : list of str, optional
        List of numeric column names to analyze. If ``None``, all numeric columns are used.
        Default is ``None``.
    method : {'pearson', 'kendall', 'spearman'}, optional
        Method of correlation. Default is ``'pearson'``.
    threshold : float, optional
        Minimum absolute correlation value to display. Default is ``0.0``.
    include_plot : bool, optional
        Whether to generate a correlation heatmap. Default is ``True``.
    figsize : tuple of int, optional
        Figure size for the heatmap. Default is ``(10, 8)``.

    Returns
    -------
    pd.DataFrame, matplotlib.figure.Figure or None
        Tuple containing:
        
        - DataFrame with correlation coefficients
        - Matplotlib Figure with the correlation heatmap,
          or ``None`` if ``include_plot=False``

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> np.random.seed(0)
    >>> df = pd.DataFrame({
    ...     'A': np.random.normal(0, 1, 100),
    ...     'B': np.random.normal(0, 1, 100),
    ...     'C': np.random.normal(0, 1, 100)
    ... })
    >>> # Create some correlation between A and B
    >>> df['B'] = df['A'] * 0.8 + df['B'] * 0.2
    >>> corr_df, fig = correlation_analysis(df, threshold=0.1)
    >>> corr_df
              A         B         C
    A  1.000000  0.832782  0.144947
    B  0.832782  1.000000  0.157413
    C  0.144947  0.157413  1.000000
    """
    # Select numeric columns
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    else:
        # Ensure all specified columns exist and are numeric
        for col in numeric_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")
            if not np.issubdtype(df[col].dtype, np.number):
                raise ValueError(f"Column '{col}' is not numeric")
    
    # Calculate correlation matrix
    corr_df = df[numeric_cols].corr(method=method)
    
    # Apply threshold if specified
    if threshold > 0:
        corr_mask = np.abs(corr_df) < threshold
        corr_df_filtered = corr_df.copy()
        corr_df_filtered[corr_mask] = np.nan
    else:
        corr_df_filtered = corr_df
    
    # Generate heatmap if requested
    fig = None
    if include_plot:
        fig, ax = plt.subplots(figsize=figsize)
        mask = np.triu(np.ones_like(corr_df, dtype=bool))
        
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(
            corr_df,
            mask=mask,
            cmap=cmap,
            vmax=1.0,
            vmin=-1.0,
            center=0,
            square=True,
            linewidths=.5,
            cbar_kws={"shrink": .8},
            annot=True,
            fmt=".2f",
            ax=ax
        )
        
        ax.set_title(f'{method.capitalize()} Correlation Matrix')
        plt.tight_layout()
    
    return corr_df_filtered, fig


def run_t_test(
    data1: Union[pd.Series, List[float], np.ndarray],
    data2: Optional[Union[pd.Series, List[float], np.ndarray]] = None,
    paired: bool = False,
    equal_var: bool = True,
    alternative: str = 'two-sided',
    include_plot: bool = True,
    figsize: Tuple[int, int] = (10, 6)
) -> Tuple[Dict[str, float], Optional[plt.Figure]]:
    """
    Perform t-test analysis between one or two data samples.

    Parameters
    ----------
    data1 : pd.Series, list, or array
        First data sample.
    data2 : pd.Series, list, or array, optional
        Second data sample. If ``None``, a one-sample t-test against 0 is performed.
        Default is ``None``.
    paired : bool, optional
        Whether to perform a paired t-test. Only valid when ``data2`` is provided.
        Default is ``False``.
    equal_var : bool, optional
        Whether to assume equal variances for the two samples. Only used for
        two-sample t-test (not paired). Default is ``True``.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Alternative hypothesis. Default is ``'two-sided'``.
    include_plot : bool, optional
        Whether to generate comparative plots. Default is ``True``.
    figsize : tuple of int, optional
        Figure size for the plots. Default is ``(10, 6)``.

    Returns
    -------
    dict, matplotlib.figure.Figure or None
        Tuple containing:
        
        - Dictionary with t-test results:
            - ``t_statistic``: The t-statistic
            - ``p_value``: The p-value
            - ``df``: Degrees of freedom
            - ``mean1``: Mean of the first sample
            - ``mean2``: Mean of the second sample (if applicable)
            - ``std1``: Standard deviation of the first sample
            - ``std2``: Standard deviation of the second sample (if applicable)
            - ``effect_size``: Cohen's d effect size

        - Matplotlib Figure with comparative plots,
          or ``None`` if ``include_plot=False``

    Examples
    --------
    >>> import numpy as np
    >>> # Two independent samples with different means
    >>> group1 = np.random.normal(10, 2, 100)
    >>> group2 = np.random.normal(12, 2, 100)
    >>> results, fig = run_t_test(group1, group2)
    >>> results['p_value'] < 0.05  # Significant difference
    True
    >>> results['mean1'] < results['mean2']  # Group 2 has higher mean
    True
    """
    # Convert inputs to numpy arrays
    data1_array = np.asarray(data1)
    
    # Calculate t-test and statistics
    results = {}
    results['mean1'] = np.mean(data1_array)
    results['std1'] = np.std(data1_array, ddof=1)
    
    # Determine the type of t-test
    if data2 is None:
        # One-sample t-test
        t_stat, p_val = stats.ttest_1samp(data1_array, 0, alternative=alternative)
        results['t_statistic'] = t_stat
        results['p_value'] = p_val
        results['df'] = len(data1_array) - 1
        results['effect_size'] = np.abs(results['mean1']) / results['std1']  # Cohen's d
        test_type = "One-sample t-test"
        
    else:
        # Two-sample t-test (paired or independent)
        data2_array = np.asarray(data2)
        results['mean2'] = np.mean(data2_array)
        results['std2'] = np.std(data2_array, ddof=1)
        
        if paired:
            # Paired t-test
            if len(data1_array) != len(data2_array):
                raise ValueError("Paired t-test requires samples of equal length")
            
            t_stat, p_val = stats.ttest_rel(data1_array, data2_array, alternative=alternative)
            results['t_statistic'] = t_stat
            results['p_value'] = p_val
            results['df'] = len(data1_array) - 1
            
            # Effect size (Cohen's d for paired samples)
            diff = data1_array - data2_array
            results['effect_size'] = np.mean(diff) / np.std(diff, ddof=1)
            test_type = "Paired t-test"
            
        else:
            # Independent two-sample t-test
            t_stat, p_val = stats.ttest_ind(
                data1_array, data2_array, 
                equal_var=equal_var, 
                alternative=alternative
            )
            results['t_statistic'] = t_stat
            results['p_value'] = p_val
            
            # Degrees of freedom
            if equal_var:
                results['df'] = len(data1_array) + len(data2_array) - 2
            else:
                # Welch-Satterthwaite equation for degrees of freedom
                var1 = results['std1']**2
                var2 = results['std2']**2
                n1 = len(data1_array)
                n2 = len(data2_array)
                results['df'] = ((var1/n1 + var2/n2)**2) / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
            
            # Effect size (Cohen's d for independent samples)
            pooled_std = np.sqrt(((len(data1_array) - 1) * results['std1']**2 + 
                                 (len(data2_array) - 1) * results['std2']**2) / 
                                (len(data1_array) + len(data2_array) - 2))
            results['effect_size'] = np.abs(results['mean1'] - results['mean2']) / pooled_std
            
            test_type = "Independent t-test" + (" (equal variances)" if equal_var else " (unequal variances)")
    
    # Generate plots if requested
    fig = None
    if include_plot:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        if data2 is None:
            # One-sample t-test visualization
            sns.histplot(data1_array, kde=True, ax=axes[0])
            axes[0].axvline(0, color='red', linestyle='--', label='Null hypothesis (μ=0)')
            axes[0].axvline(results['mean1'], color='green', linestyle='-', label='Sample mean')
            axes[0].set_title('Distribution of Sample')
            axes[0].legend()
            
            # QQ plot
            stats.probplot(data1_array, plot=axes[1])
            axes[1].set_title('Q-Q Plot')
            
        else:
            # Two-sample t-test visualization
            if paired:
                # Paired t-test - show differences
                diff = data1_array - data2_array
                sns.histplot(diff, kde=True, ax=axes[0])
                axes[0].axvline(0, color='red', linestyle='--', label='Null hypothesis (μ=0)')
                axes[0].axvline(np.mean(diff), color='green', linestyle='-', label='Mean difference')
                axes[0].set_title('Distribution of Paired Differences')
                axes[0].legend()
                
                # Scatter plot of paired samples
                axes[1].scatter(data1_array, data2_array, alpha=0.6)
                min_val = min(np.min(data1_array), np.min(data2_array))
                max_val = max(np.max(data1_array), np.max(data2_array))
                axes[1].plot([min_val, max_val], [min_val, max_val], 'r--')
                axes[1].set_xlabel('Sample 1')
                axes[1].set_ylabel('Sample 2')
                axes[1].set_title('Paired Samples')
                
            else:
                # Independent samples - show distributions
                sns.histplot(data1_array, kde=True, ax=axes[0], color='blue', label='Sample 1')
                sns.histplot(data2_array, kde=True, ax=axes[0], color='orange', label='Sample 2')
                axes[0].axvline(results['mean1'], color='blue', linestyle='--', label='Mean 1')
                axes[0].axvline(results['mean2'], color='orange', linestyle='--', label='Mean 2')
                axes[0].set_title('Distribution Comparison')
                axes[0].legend()
                
                # Box plot comparison
                data_to_plot = [data1_array, data2_array]
                axes[1].boxplot(data_to_plot, labels=['Sample 1', 'Sample 2'])
                axes[1].set_title('Box Plot Comparison')
        
        plt.suptitle(f"{test_type}\np={results['p_value']:.4f}, t={results['t_statistic']:.4f}", fontsize=12)
        plt.tight_layout()
    
    return results, fig


def time_series_analysis(
    time_series: pd.Series,
    periods: int = 10,
    seasonal: bool = True,
    seasonal_periods: int = 12,
    include_plots: bool = True,
    figsize: Tuple[int, int] = (12, 10)
) -> Tuple[Dict[str, Union[pd.Series, pd.DataFrame]], Optional[plt.Figure]]:
    """
    Perform basic time series analysis and forecasting.

    Parameters
    ----------
    time_series : pd.Series
        Time series data with a DatetimeIndex.
    periods : int, optional
        Number of periods to forecast. Default is ``10``.
    seasonal : bool, optional
        Whether to include seasonal components in the decomposition.
        Default is ``True``.
    seasonal_periods : int, optional
        Number of periods in a seasonal cycle. Default is ``12`` (monthly data).
    include_plots : bool, optional
        Whether to generate analysis plots. Default is ``True``.
    figsize : tuple of int, optional
        Figure size for the plots. Default is ``(12, 10)``.

    Returns
    -------
    dict, matplotlib.figure.Figure or None
        Tuple containing:
        
        - Dictionary with time series analysis results:
            - ``decomposition``: Seasonal decomposition of the time series
            - ``rolling_statistics``: Rolling mean and standard deviation
            - ``acf``: Autocorrelation function values
            - ``pacf``: Partial autocorrelation function values
            - ``forecast``: Simple forecast based on moving average
        
        - Matplotlib Figure with analysis plots,
          or ``None`` if ``include_plots=False``

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Create a simple time series with trend and seasonality
    >>> dates = pd.date_range(start='2020-01-01', periods=100, freq='M')
    >>> trend = np.arange(100) * 0.1
    >>> seasonality = np.sin(np.arange(100) * 2 * np.pi / 12) * 5
    >>> noise = np.random.normal(0, 1, 100)
    >>> data = trend + seasonality + noise
    >>> ts = pd.Series(data, index=dates)
    >>> results, fig = time_series_analysis(ts)
    >>> results['decomposition'].trend.head()
    2020-01-31    0.100005
    2020-02-29    0.200010
    2020-03-31    0.300015
    2020-04-30    0.400019
    2020-05-31    0.500024
    Freq: M, Name: trend, dtype: float64
    """
    # Validate input
    if not isinstance(time_series, pd.Series):
        raise TypeError("Input must be a pandas Series")
    if not isinstance(time_series.index, pd.DatetimeIndex):
        raise TypeError("Series index must be a DatetimeIndex")
    
    results = {}
    
    # Perform seasonal decomposition if there are enough data points
    if len(time_series) >= 2 * seasonal_periods and seasonal:
        decomposition = stats.seasonal_decompose(
            time_series, 
            model='additive', 
            period=seasonal_periods
        )
        results['decomposition'] = decomposition
    else:
        seasonal = False  # Not enough data for seasonal decomposition
    
    # Calculate rolling statistics
    window_size = min(len(time_series) // 4, 12)  # Use 1/4 of data points or 12, whichever is smaller
    rolling_mean = time_series.rolling(window=window_size).mean()
    rolling_std = time_series.rolling(window=window_size).std()
    results['rolling_statistics'] = pd.DataFrame({
        'original': time_series,
        'rolling_mean': rolling_mean,
        'rolling_std': rolling_std
    })
    
    # Calculate autocorrelation and partial autocorrelation
    max_lags = min(len(time_series) - 1, 24)  # Use up to 24 lags or n-1, whichever is smaller
    acf_values = stats.acf(time_series.dropna(), nlags=max_lags)
    pacf_values = stats.pacf(time_series.dropna(), nlags=max_lags)
    results['acf'] = pd.Series(acf_values, index=range(len(acf_values)))
    results['pacf'] = pd.Series(pacf_values, index=range(len(pacf_values)))
    
    # Simple forecasting using moving average
    ma_window = min(len(time_series) // 4, 12)  # Use 1/4 of data points or 12, whichever is smaller
    ma_forecast = time_series.rolling(window=ma_window).mean().iloc[-1]
    
    # Create forecast Series
    last_date = time_series.index[-1]
    forecast_dates = pd.date_range(start=last_date, periods=periods+1, freq=time_series.index.freq)[1:]
    forecast_values = np.full(periods, ma_forecast)
    forecast = pd.Series(forecast_values, index=forecast_dates)
    results['forecast'] = forecast
    
    # Generate plots if requested
    fig = None
    if include_plots:
        if seasonal and 'decomposition' in results:
            fig, axes = plt.subplots(5, 1, figsize=figsize, sharex=True)
            
            # Original time series
            time_series.plot(ax=axes[0], title='Original Time Series')
            axes[0].grid(True, linestyle='--', alpha=0.7)
            
            # Trend component
            results['decomposition'].trend.plot(ax=axes[1], title='Trend')
            axes[1].grid(True, linestyle='--', alpha=0.7)
            
            # Seasonal component
            results['decomposition'].seasonal.plot(ax=axes[2], title='Seasonality')
            axes[2].grid(True, linestyle='--', alpha=0.7)
            
            # Residual component
            results['decomposition'].resid.plot(ax=axes[3], title='Residuals')
            axes[3].grid(True, linestyle='--', alpha=0.7)
            
            # Forecast
            ax_forecast = axes[4]
        else:
            fig, axes = plt.subplots(3, 1, figsize=(figsize[0], figsize[1] * 3/5), sharex=True)
            
            # Original time series with rolling statistics
            ax_ts = axes[0]
            time_series.plot(ax=ax_ts, label='Original')
            rolling_mean.plot(ax=ax_ts, label='Rolling Mean', color='red')
            rolling_std.plot(ax=ax_ts, label='Rolling Std', color='green')
            ax_ts.set_title('Time Series with Rolling Statistics')
            ax_ts.grid(True, linestyle='--', alpha=0.7)
            ax_ts.legend()
            
            # ACF and PACF
            lags = np.arange(len(results['acf']))
            axes[1].bar(lags, results['acf'], width=0.3)
            axes[1].set_title('Autocorrelation Function')
            axes[1].grid(True, linestyle='--', alpha=0.7)
            
            axes[2].bar(lags, results['pacf'], width=0.3)
            axes[2].set_title('Partial Autocorrelation Function')
            axes[2].grid(True, linestyle='--', alpha=0.7)
            
            # Create a new axis for forecast
            fig.add_subplot(4, 1, 4)
            ax_forecast = plt.gca()
        
        # Plot forecast
        time_series.plot(ax=ax_forecast, label='Original', color='blue')
        forecast.plot(ax=ax_forecast, label='Forecast', color='red', linestyle='--')
        ax_forecast.set_title('Forecast')
        ax_forecast.grid(True, linestyle='--', alpha=0.7)
        ax_forecast.legend()
        
        plt.tight_layout()
    
    return results, fig


if __name__ == "__main__":
    # Example usage of the module
    import pandas as pd
    import numpy as np
    
    # Create a sample time series dataset
    dates = pd.date_range(start='2020-01-01', periods=36, freq='M')
    trend = np.arange(36) * 0.2
    seasonality = np.sin(np.arange(36) * 2 * np.pi / 12) * 5
    noise = np.random.normal(0, 1, 36)
    ts_data = trend + seasonality + noise
    time_series = pd.Series(ts_data, index=dates)
    
    print("Time Series Analysis Example:")
    results, _ = time_series_analysis(time_series)
    
    print("\nForecast for next 5 periods:")
    print(results['forecast'].head())
    
    # Create a sample dataset for other analyses
    np.random.seed(0)
    data = {
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(5, 2, 100),
        'feature3': np.random.exponential(2, 100)
    }
    df = pd.DataFrame(data)
    
    # Add some correlation
    df['feature4'] = df['feature1'] * 1.5 + df['feature2'] * 0.5 + np.random.normal(0, 1, 100)
    
    print("\nDescriptive Statistics:")
    stats_df, _ = descriptive_stats(df)
    print(stats_df.round(2))
    
    print("\nCorrelation Analysis:")
    corr_df, _ = correlation_analysis(df)
    print(corr_df.round(2))