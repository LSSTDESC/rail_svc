"""Top level for python client API"""

from __future__ import annotations

from typing import Any

import httpx

from .clientconfig import client_config


class RailClient:
    """Interface for accessing remote cm-service."""

    def __init__(self) -> None:
        client_kwargs: dict[str, Any] = {}
        client_kwargs["base_url"] = client_config.service_url
        client_kwargs.update(**self._extra_client_kwargs())
        self._client = httpx.Client(**client_kwargs)

    @property
    def client(self) -> httpx.Client:
        """Return the httpx.Client"""
        return self._client

    def _extra_client_kwargs(self) -> dict:  # pragma: no cover
        client_kwargs: dict[str, Any] = {}
        if "auth_token" in client_config.model_fields_set:
            client_kwargs["headers"] = {"Authorization": f"Bearer {client_config.auth_token}"}
        if "timeout" in client_config.model_fields_set:
            client_kwargs["timeout"] = client_config.timeout
        if "cookies" in client_config.model_fields_set:
            cookies = httpx.Cookies()
            if client_config.cookies:
                for cookie in client_config.cookies:
                    cookies.set(name=cookie.name, value=cookie.value)
            client_kwargs["cookies"] = cookies
        return client_kwargs


class ClientBase:
    """Interface for accessing remote rail-svc"""

    def __init__(self, parent: RailClient) -> None:
        self._client = parent.client

    @property
    def client(self) -> httpx.Client:
        """Return the httpx.Client"""
        return self._client
