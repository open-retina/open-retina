"""Tests for ``openretina.utils.misc``."""

import pytest
import requests

from openretina.utils.misc import check_server_responding

URL = "https://example.invalid/"


class _Response:
    def __init__(self, status_code: int):
        self.status_code = status_code


@pytest.mark.parametrize(
    "raised",
    [
        # requests' own ConnectionError, which is NOT a subclass of the builtin of the same name --
        # `except ConnectionError` used to let this escape and abort pytest collection.
        requests.exceptions.ConnectionError("unreachable"),
        requests.exceptions.Timeout("too slow"),
        requests.exceptions.TooManyRedirects("loop"),
    ],
)
def test_check_server_responding_is_false_on_network_errors(monkeypatch, raised: Exception) -> None:
    def _raise(*args, **kwargs):
        raise raised

    monkeypatch.setattr(requests, "get", _raise)
    assert check_server_responding(URL) is False


@pytest.mark.parametrize("status_code, expected", [(200, True), (404, False), (500, False)])
def test_check_server_responding_follows_status_code(monkeypatch, status_code: int, expected: bool) -> None:
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: _Response(status_code))
    assert check_server_responding(URL) is expected


def test_check_server_responding_passes_a_timeout(monkeypatch) -> None:
    """Without a timeout an unresponsive host stalls pytest collection instead of failing fast."""
    captured: dict = {}

    def _capture(url, **kwargs):
        captured.update(url=url, **kwargs)
        return _Response(200)

    monkeypatch.setattr(requests, "get", _capture)
    assert check_server_responding(URL, timeout=1.5) is True
    assert captured["timeout"] == 1.5
