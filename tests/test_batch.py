import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from dbutils_batch_query.model_query import batch_model_query

# test_model_query.py

"""
=======================================================
Unit Tests for Model Query Utilities
=======================================================

This module contains unit tests for the ``batch_model_query`` function in the
``model_query`` module. The tests use mocking to simulate API calls and secret retrieval,
ensuring that the batch processing logic, concurrency, and result formatting are correct
without requiring real API credentials or network access.

.. currentmodule:: model_query

Tested Functions
================

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: function_name_only.rst

    batch_model_query

"""


@pytest.mark.asyncio
async def test_batch_model_query_basic(monkeypatch):
    """
    Test ``batch_model_query`` basic functionality with a mocked OpenAI client and secrets.

    Parameters
    ----------
    monkeypatch : fixture
        Pytest fixture for patching modules and attributes.

    Returns
    -------
    None

    .. Note::
        This test mocks both the OpenAI client and the Databricks secrets retrieval.
    """

    # Mock get_databricks_secrets to return dummy token and host
    monkeypatch.setattr(
        "dbutils_batch_query.model_query.get_databricks_secrets",
        lambda: ("dummy_token", "https://dummy_host"),
    )

    # Prepare a fake chat completion response
    class FakeUsage:
        prompt_tokens = 5
        completion_tokens = 7
        total_tokens = 12

    class FakeChoice:
        message = MagicMock(content="Hello, world!")

    class FakeChatCompletion:
        model = "databricks-llama-4-maverick"
        usage = FakeUsage()
        choices = [FakeChoice()]

    # Patch AsyncOpenAI to return a mock client with a mock chat.completions.create
    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(return_value=FakeChatCompletion())

    monkeypatch.setattr(
        "dbutils_batch_query.model_query.AsyncOpenAI",
        lambda api_key, base_url: fake_client,
    )

    # Patch tqdm.gather to just use asyncio.gather (no progress bar in test)
    monkeypatch.setattr("dbutils_batch_query.model_query.tqdm.gather", asyncio.gather)

    # Prepare test prompts
    prompts = [
        {"system": "You are a helpful assistant.", "user": "Say hello."},
        {"system": "You are a helpful assistant.", "user": "Say goodbye."},
    ]

    # Run batch_model_query
    results = await batch_model_query(
        prompt_info=prompts,
        model="databricks-llama-4-maverick",
        batch_size=1,
        max_concurrent_requests=2,
    )

    # Check results
    assert isinstance(results, list)
    assert len(results) == 2
    for result in results:
        assert result["message"] == "Hello, world!"
        assert result["model"] == "databricks-llama-4-maverick"
        assert result["prompt_tokens"] == 5
        assert result["completion_tokens"] == 7
        assert result["total_tokens"] == 12
        assert result["error"] is None
        assert "system" in result and "user" in result


@pytest.mark.asyncio
async def test_batch_model_query_with_process_func(monkeypatch):
    """
    Test ``batch_model_query`` with a custom process_func and mocked dependencies.

    Parameters
    ----------
    monkeypatch : fixture
        Pytest fixture for patching modules and attributes.

    Returns
    -------
    None

    .. Note::
        This test checks that the process_func is called and its output is included.
    """

    monkeypatch.setattr(
        "dbutils_batch_query.model_query.get_databricks_secrets",
        lambda: ("dummy_token", "https://dummy_host"),
    )

    class FakeUsage:
        prompt_tokens = 2
        completion_tokens = 3
        total_tokens = 5

    class FakeChoice:
        message = MagicMock(content="42")

    class FakeChatCompletion:
        model = "databricks-llama-4-maverick"
        usage = FakeUsage()
        choices = [FakeChoice()]

    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(return_value=FakeChatCompletion())

    monkeypatch.setattr(
        "dbutils_batch_query.model_query.AsyncOpenAI",
        lambda api_key, base_url: fake_client,
    )
    monkeypatch.setattr("dbutils_batch_query.model_query.tqdm.gather", asyncio.gather)

    prompts = [{"system": "You are a calculator.", "user": "What is 6 * 7?"}]

    def process_func(text: str) -> dict:
        return {"answer": int(text)}

    results = await batch_model_query(
        prompt_info=prompts,
        model="databricks-llama-4-maverick",
        process_func=process_func,
        batch_size=1,
        max_concurrent_requests=1,
    )

    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]["processed_response"] == {"answer": 42}
    assert results[0]["message"] == "42"
    assert results[0]["model"] == "databricks-llama-4-maverick"
    assert results[0]["error"] is None


@pytest.mark.asyncio
async def test_batch_model_query_error(monkeypatch):
    """
    Test ``batch_model_query`` error handling when the API call fails.

    Parameters
    ----------
    monkeypatch : fixture
        Pytest fixture for patching modules and attributes.

    Returns
    -------
    None

    .. Note::
        This test simulates an exception in the API call and checks error reporting.
    """

    monkeypatch.setattr(
        "dbutils_batch_query.model_query.get_databricks_secrets",
        lambda: ("dummy_token", "https://dummy_host"),
    )

    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(
        side_effect=RuntimeError("API error")
    )

    monkeypatch.setattr(
        "dbutils_batch_query.model_query.AsyncOpenAI",
        lambda api_key, base_url: fake_client,
    )
    monkeypatch.setattr("dbutils_batch_query.model_query.tqdm.gather", asyncio.gather)

    prompts = [{"system": "You are a helpful assistant.", "user": "Trigger error."}]

    results = await batch_model_query(
        prompt_info=prompts, model="gpt-err", batch_size=1, max_concurrent_requests=1
    )

    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]["message"] is None
    assert results[0]["error"] is not None
    assert "API error" in results[0]["error"]
