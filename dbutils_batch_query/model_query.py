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
Model Query Utilities
=======================================================

This module provides functionality to interact with language models, process responses,
and handle batch operations asynchronously for AI-powered knowledge extraction.

Model Query Processing Pipeline
===========================================================

The query processing pipeline consists of the following stages:

1. **Preparation**:

   - Format prompts with system and user contexts
   - Configure batch processing parameters

2. **Asynchronous Execution**:

   - Send requests to language models with rate limiting
   - Track execution time and token usage metrics

3. **Response Processing**:

   - Extract structured information from model responses
   - Apply custom processing functions to raw outputs
   - Handle errors gracefully with default return values

4. **Result Management**:

   - Save intermediate and final results
   - Provide comprehensive metadata for analysis

.. Note::
    - This module is designed for efficient batch processing of multiple queries.
    - All API interactions are performed asynchronously to maximize throughput.

.. Important::
    - Ensure proper API credentials are configured before using this module.
    - Consider token usage and rate limits when processing large batches.

.. currentmodule:: dbutils_batch_query.model_query

Functions
=========

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: function_name_only.rst

    with_default_return
    extract_json_items
    batch_model_query

Standalone Execution
=====================
This module is not intended to be run as a standalone script.

"""

__author__ = "William Muntean"
__email__ = "williamjmuntean@gmail.com"
__license__ = "GPL v3"
__maintainer__ = "William Muntean"
__date__ = "2025-03-13"

import asyncio
import functools
import json
import pickle
import re
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, TypeVar, cast

import pandas as pd
from aiolimiter import AsyncLimiter
from json_repair import repair_json
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

from .utils.secrets import get_databricks_secrets

F = TypeVar("F", bound=Callable[..., Dict[str, Any]])


def with_default_return(default_return: Dict[str, Any]) -> Callable[[F], F]:
    """
    Decorator to add a default_return attribute to a function.

    This decorator adds a ``default_return`` attribute to the decorated function,
    which can be used for error handling to provide properly structured returns
    when the function cannot execute normally.

    Parameters
    ----------
    default_return : dict
        Dictionary containing the default structure to be returned in error cases.
        Keys should match the expected return structure of the decorated function.

    Returns
    -------
    callable
        Decorator function that attaches the default_return attribute to the
        target function.

    .. Note::
        This decorator preserves the original function's signature, docstring,
        and other attributes.

    Examples
    --------
    >>> @with_default_return(
    ...     {"tagged_items": [], "non_tagged_items": []}
    ... )
    ... def extract_items(content):
    ...     # Function implementation
    ...     return {
    ...         "tagged_items": ["item1"],
    ...         "non_tagged_items": ["item2"],
    ...     }
    >>> extract_items.default_return
    {'tagged_items': [], 'non_tagged_items': []}
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.default_return = default_return
        return cast(F, wrapper)

    return decorator


@with_default_return([])
def extract_json_items(text: str) -> dict:
    """
    Extracts and parses all JSON objects or arrays from code blocks in the input text.

    Parameters
    ----------
    text : str
        The input text containing JSON enclosed within triple backticks with json syntax highlighting.

    Returns
    -------
    dict
        A dictionary with an ``items`` key containing a list of all parsed JSON objects.
        Returns an empty list if no valid JSON is found or parsing fails.

    .. Note::
        If a JSON code block is invalid, that specific block will be skipped but
        the function will continue processing other valid JSON code blocks.

    .. Warning::
        Ensure that the input text contains properly formatted JSON in code blocks.
        The ``repair_json`` package is used to help support noncompliant JSON.

    Examples
    --------
    >>> text = '''
    Some intro text.
    ```json
    {
        "key1": "value1",
        "key2": "value2"
    }
    ```
    More text between code blocks.
    ```json
    {
        "key3": "value3",
        "key4": "value4"
    }
    ```
    '''
    >>> extract_json_items(text)
    {'items': [{'key1': 'value1', 'key2': 'value2'}, {'key3': 'value3', 'key4': 'value4'}]}
    """
    # Regular expression to extract all content within triple backticks
    code_block_pattern = re.compile(r"```json(.*?)```", re.DOTALL)
    matches = code_block_pattern.findall(text)

    items = []

    for match in matches:
        try:
            # Attempt to parse the matched text as JSON
            match = repair_json(match.strip())
            json_content = json.loads(match.strip())
            if isinstance(json_content, list):
                items.extend(json_content)
            else:
                items.append(json_content)
        except json.JSONDecodeError:
            # Skip invalid JSON
            continue

    return items


async def _get_response(
    prompt_info: dict[str, str],
    client: AsyncOpenAI,
    model: str,
    process_func: Callable[[str], dict] | None,
    semaphore: asyncio.Semaphore,
    rate_limiter: AsyncLimiter,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Process a single prompt and return the model's response with detailed metadata.

    Parameters
    ----------
    prompt_info : dict[str, str]
        Dictionary containing prompt information including system prompt and user prompt.
        May contain additional fields that will be preserved and included in the return dictionary.
    client : AsyncOpenAI
        The asynch OpenAI client instance.
    model : str
        The language model identifier.
    process_func : callable or None
        Function to process the response content. If None, raw content is returned.
    semaphore : asyncio.Semaphore
        Semaphore for limiting concurrent requests.
    rate_limiter : AsyncLimiter
        Rate limiter to control the number of requests per time interval. The rate limit is enforced
        for each API call to avoid exceeding provider quotas or triggering throttling.
    model_params : dict, optional
        Dictionary of model parameters to override defaults. Supported keys:
        - ``max_tokens``: Maximum tokens for completion (default: 2048)
        - ``temperature``: Sampling temperature (default: 0)

    Returns
    -------
    dict
        A flat dictionary containing all the following fields:

        - ``message``: Raw response content from the model
        - ``processed_response``: Present if process_func is provided, containing processed content
        - ``chat``: Full API response object
        - ``error``: Error message if an exception occurred, None otherwise
        - ``model``: Model name used for generation
        - ``temperature``: Temperature setting used for generation
        - ``max_tokens``: Maximum tokens setting used
        - ``prompt_tokens``: Number of tokens in the prompt
        - ``completion_tokens``: Number of tokens in the completion
        - ``total_tokens``: Total number of tokens used
        - ``timing``: Query execution time in seconds
        - All original keys from prompt_info (including system and prompt)

    .. Note::
        The function measures and reports execution time for the model query.
        All metadata fields and prompt_info fields are merged into the response
        dictionary to create a flat structure.

    .. Important::
        The rate limiter is used to ensure that API requests do not exceed the allowed
        rate, as specified by the ``rate_limit`` parameter in the batch processing function.
    """

    # Default model parameters
    DEFAULT_MAX_TOKENS = 2048
    DEFAULT_TEMPERATURE = 0

    # Use provided model parameters or defaults
    model_params = model_params or {}
    max_tokens = model_params.get("max_tokens", DEFAULT_MAX_TOKENS)
    temperature = model_params.get("temperature", DEFAULT_TEMPERATURE)

    async with semaphore:
        async with rate_limiter:
            # Initialize metadata dictionary
            metadata = {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "usage": None,
                "timing": None,
            }

            start_time = time.perf_counter()
            response = {"start": start_time}

            try:
                messages = [
                    {"role": "system", "content": prompt_info["system"]},
                    {"role": "user", "content": prompt_info["user"]},
                ]

                chat_completion = await client.chat.completions.create(
                    messages=messages,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                # Calculate and record execution time
                execution_time = time.perf_counter() - start_time
                metadata["timing"] = round(execution_time, 3)

                # Update model info with version number
                if hasattr(chat_completion, "model"):
                    metadata["model"] = chat_completion.model

                # Extract usage information if available
                if hasattr(chat_completion, "usage") and chat_completion.usage:
                    metadata["prompt_tokens"] = getattr(
                        chat_completion.usage, "prompt_tokens", None
                    )
                    metadata["completion_tokens"] = getattr(
                        chat_completion.usage, "completion_tokens", None
                    )
                    metadata["total_tokens"] = getattr(
                        chat_completion.usage, "total_tokens", None
                    )

                response = {"message": chat_completion.choices[0].message.content}
                if process_func:
                    response["processed_response"] = process_func(
                        chat_completion.choices[0].message.content
                    )

                response["chat"] = chat_completion
                response["error"] = None

            except Exception as e:
                # Calculate execution time even for failed requests
                execution_time = time.perf_counter() - start_time
                metadata["timing"] = execution_time

                response = {"message": None}
                if process_func:
                    response["processed_response"] = getattr(
                        process_func, "default_return", {}
                    )
                response["chat"] = None
                response["error"] = str(e)

            response |= metadata
            response |= prompt_info
            return response


async def batch_model_query(
    *,
    prompt_info: list[dict],
    model: str,
    process_func: callable = None,
    process_func_params: dict = None,
    batch_size: int = 10,
    max_concurrent_requests: int = 5,
    rate_limit: tuple = (2, 1),
    results_path: str | Path = None,
    run_name: str | Path = None,
    model_params: dict = None,
    token: str = None,
    host: str = None,
    **kwargs,
) -> list[dict]:
    """
    Asynchronously process a batch of prompts using a language model client, saving intermediate and final results,
    and handling concurrent API requests with robust error handling and metadata tracking.

    This function orchestrates the batch processing pipeline for querying language models. It divides the input prompts
    into batches, manages concurrency, applies optional post-processing, and saves results in both pickle and parquet formats.
    Intermediate results are saved after each batch for fault tolerance, and final results are consolidated at the end.

    Parameters
    ----------
    prompt_info : list of dict
        List of dictionaries containing prompt information, each with ``system`` and ``user`` keys.
    model : str
        The name or identifier of the language model to use.
    process_func : callable, optional
        Function to process each response. If None, raw message content is returned in the ``message`` field.
        Default is ``None``.
    process_func_params : dict, optional
        Parameters to pass to ``process_func``. Ignored if ``process_func`` is None. Default is ``None``.
    batch_size : int, optional
        Number of prompts to process between intermittent saves. Default is ``10``.
    max_concurrent_requests : int, optional
        Maximum number of concurrent API requests allowed. Default is ``5``.
    rate_limit : tuple, optional
        Tuple of (max_requests, interval_seconds) specifying the maximum number of requests allowed per interval.
        Default is ``(2, 1)``, meaning 2 requests per 1 second. This is enforced using a rate limiter to avoid
        exceeding API provider quotas or triggering throttling.
    results_path : str or Path, optional
        Path to save intermediate and final result files. If None, results are not saved. Default is ``None``.
    run_name : str or Path, optional
        Name used to identify this batch run in saved files. Required if ``results_path`` is provided. Default is ``None``.
    model_params : dict, optional
        Dictionary of model parameters to override defaults. Supported keys:
        - ``max_tokens``: Maximum tokens for completion (default: 2048)
        - ``temperature``: Sampling temperature (default: 0)
        Default is ``None``.
    token : str, optional
        API token for authentication. If not provided, will be loaded from environment or Databricks context.
    host : str, optional
        API host URL. If not provided, will be loaded from environment or Databricks context.

    Returns
    -------
    list of dict
        A list of dictionaries, each containing the response and associated metadata for a prompt. Each dictionary includes:

        - ``message``: Raw response content from the model.
        - ``processed_response``: Processed content if ``process_func`` is provided.
        - ``chat``: Full API response object (or ``None`` on error).
        - ``error``: Error message if an exception occurred, ``None`` otherwise.
        - ``model``: Model name used for generation.
        - ``temperature``: Temperature setting used for generation.
        - ``max_tokens``: Maximum tokens setting used.
        - ``prompt_tokens``: Number of tokens in the prompt (if available).
        - ``completion_tokens``: Number of tokens in the completion (if available).
        - ``total_tokens``: Total number of tokens used (if available).
        - ``timing``: Query execution time in seconds.
        - All original keys from the corresponding entry in ``prompt_info``.

    .. Note::
        - When ``process_func`` is None, the function returns the raw message content in the ``message`` field.
        - After each batch, results (including ``chat`` objects) are saved as pickle files, and a version without the ``chat`` key is saved as a parquet file.
        - Intermediate results are saved in a subdirectory named after ``run_name``; final results are saved in ``results_path``.
        - Intermediate files are deleted after successful completion.

    .. Warning::
        - Ensure that ``token`` and ``host`` are set, either via arguments, environment variables, env dotfile, or Databricks context.
        - If ``results_path`` is provided, ``run_name`` must also be specified.

    Examples
    --------
    .. include:: ../../../../README.md
        :parser: myst_parser.sphinx_
        :start-after: <!-- start Example -->
        :end-before: <!-- end Example -->
    """

    if not token and not host:
        token, host = get_databricks_secrets()

    client = AsyncOpenAI(
        api_key=token,
        base_url=f"{host}/serving-endpoints",
    )

    wrapped_process_func = None
    if process_func is not None and process_func_params is not None:
        wrapped_process_func = functools.partial(process_func, **process_func_params)
    elif process_func is not None:
        wrapped_process_func = process_func

    semaphore = asyncio.Semaphore(max_concurrent_requests)

    rate_limiter = AsyncLimiter(*rate_limit)
    all_results = []

    num_prompts = len(prompt_info)
    for start in range(0, num_prompts, batch_size):
        end = min(start + batch_size, num_prompts)
        prompt_batch = prompt_info[start:end]

        tasks = [
            _get_response(
                prompt,
                client,
                model,
                wrapped_process_func,
                semaphore,
                rate_limiter,
                model_params,
            )
            for prompt in prompt_batch
        ]

        batch_results = await tqdm.gather(*tasks)
        all_results.extend(batch_results)

        # Save intermediate results if paths are provided
        if results_path and run_name:
            # Create results_path directory if it doesn't exist
            if isinstance(results_path, str):
                results_path = Path(results_path)
            inter_path = results_path / run_name
            inter_path.mkdir(parents=True, exist_ok=True)

            # Save all_results directly as pickle
            pickle_path = inter_path / f"{run_name}.pkl"
            with open(pickle_path, "wb") as f:
                pickle.dump(all_results, f)

            # Create a copy of results without 'chat' key for parquet
            results_no_chat = []
            for result in all_results:
                result_copy = result.copy()
                if "chat" in result_copy:
                    del result_copy["chat"]
                results_no_chat.append(pd.DataFrame(result_copy))

            # Save as parquet
            df_batch = pd.concat(results_no_chat)
            df_batch.to_parquet(inter_path / f"{run_name}.parquet", index=False)

    if results_path and run_name:
        # Save all_results directly as pickle
        pickle_path = results_path / f"{run_name}.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(all_results, f)

        # Create a copy of results without 'chat' key for parquet
        results_no_chat = []
        for result in all_results:
            result_copy = result.copy()
            if "chat" in result_copy:
                del result_copy["chat"]
            results_no_chat.append(pd.DataFrame(result_copy))

        # Save as parquet
        df_batch = pd.concat(results_no_chat)
        df_batch.to_parquet(results_path / f"{run_name}.parquet", index=False)

        # Delete intermediate results if requested
        if inter_path.exists():
            shutil.rmtree(inter_path)

    return all_results
