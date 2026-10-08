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
        # Not a subclass of the builtin ConnectionError.
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


def test_check_server_responding_passes_a_timeout(monkeypatch) -> None:
    """Without a timeout an unresponsive host stalls pytest collection instead of failing fast."""
    captured: dict = {}

    def _capture(url, **kwargs):
        captured.update(url=url, **kwargs)
        return _Response(200)

    monkeypatch.setattr(requests, "get", _capture)
    assert check_server_responding(URL, timeout=1.5) is True
    assert captured["timeout"] == 1.5


@pytest.mark.parametrize("bad_url", ["example.invalid/no-scheme", "https://"])
def test_check_server_responding_propagates_malformed_urls(bad_url: str) -> None:
    """A typo in the URL is a caller bug, not an unreachable server."""
    with pytest.raises(requests.exceptions.RequestException):
        check_server_responding(bad_url)
