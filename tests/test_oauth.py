"""
Unit tests for oauth.py.

No test in this file makes a real network call or writes to a real
~/.robinhood-agent — urllib.request.urlopen is monkeypatched wherever a
token-endpoint request would otherwise happen, and every save/load test
uses tmp_path instead of oauth.DEFAULT_TOKENS_PATH.
"""

import json
import os
import stat
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import oauth
from oauth import (
    DEFAULT_TOKENS_PATH,
    OAuthError,
    OAuthStateMismatchError,
    _check_callback,
    _needs_refresh,
    generate_pkce,
    get_valid_access_token,
    load_tokens,
    save_tokens,
)


class _FakeHTTPResponse:
    """Stands in for the object urllib.request.urlopen()'s context manager
    yields — just enough surface for oauth._token_request()'s use of it."""

    def __init__(self, body: dict):
        self._body = json.dumps(body).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestGeneratePkce:
    def test_challenge_matches_the_rfc_7636_appendix_b_vector(self, monkeypatch):
        # https://www.rfc-editor.org/rfc/rfc7636#appendix-B — the reference
        # verifier/challenge pair from the spec itself, not a value this
        # test invented.
        known_verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
        known_challenge = "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"

        monkeypatch.setattr(oauth.secrets, "token_urlsafe", lambda n: known_verifier)

        verifier, challenge = generate_pkce()

        assert verifier == known_verifier
        assert challenge == known_challenge

    def test_verifier_length_is_within_rfc_7636_bounds(self):
        verifier, _ = generate_pkce()
        assert 43 <= len(verifier) <= 128

    def test_challenge_has_no_padding_characters(self):
        _, challenge = generate_pkce()
        assert "=" not in challenge


class TestCheckCallback:
    def test_matching_state_returns_the_code(self):
        code = _check_callback({"state": "abc123", "code": "the-code"}, expected_state="abc123")
        assert code == "the-code"

    def test_state_mismatch_is_rejected(self):
        with pytest.raises(OAuthStateMismatchError):
            _check_callback({"state": "wrong", "code": "the-code"}, expected_state="abc123")

    def test_missing_state_is_rejected(self):
        with pytest.raises(OAuthStateMismatchError):
            _check_callback({"code": "the-code"}, expected_state="abc123")

    def test_authorization_server_error_is_raised(self):
        with pytest.raises(OAuthError):
            _check_callback(
                {"state": "abc123", "error": "access_denied"}, expected_state="abc123"
            )

    def test_missing_code_is_rejected(self):
        with pytest.raises(OAuthError):
            _check_callback({"state": "abc123"}, expected_state="abc123")


class TestTokenStorage:
    def test_default_tokens_path_is_outside_the_repo(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        resolved = os.path.abspath(DEFAULT_TOKENS_PATH)

        assert not resolved.startswith(repo_root)

    def test_save_then_load_round_trips(self, tmp_path):
        path = str(tmp_path / "tokens.json")
        tokens = {"access_token": "a", "refresh_token": "r", "expires_at": 123.0}

        save_tokens(tokens, path=path)
        loaded = load_tokens(path=path)

        assert loaded == tokens

    def test_load_returns_none_when_file_does_not_exist(self, tmp_path):
        path = str(tmp_path / "does-not-exist.json")
        assert load_tokens(path=path) is None

    def test_save_creates_missing_parent_directory(self, tmp_path):
        path = str(tmp_path / "nested" / "dir" / "tokens.json")
        save_tokens({"access_token": "a"}, path=path)
        assert os.path.exists(path)

    @pytest.mark.skipif(
        os.name != "posix", reason="POSIX permission bits don't apply on Windows"
    )
    def test_saved_file_is_owner_only(self, tmp_path):
        path = str(tmp_path / "tokens.json")
        save_tokens({"access_token": "a"}, path=path)

        mode = stat.S_IMODE(os.stat(path).st_mode)
        assert mode == stat.S_IRUSR | stat.S_IWUSR


class TestNeedsRefresh:
    def test_far_from_expiry_does_not_need_refresh(self):
        tokens = {"expires_at": 1000.0}
        assert _needs_refresh(tokens, now=100.0, leeway=60) is False

    def test_within_leeway_of_expiry_needs_refresh(self):
        tokens = {"expires_at": 1000.0}
        assert _needs_refresh(tokens, now=941.0, leeway=60) is True

    def test_exactly_at_the_leeway_boundary_needs_refresh(self):
        # "expires within 60 seconds" is inclusive of exactly 60.
        tokens = {"expires_at": 1000.0}
        assert _needs_refresh(tokens, now=940.0, leeway=60) is True

    def test_one_second_before_the_boundary_does_not_need_refresh(self):
        tokens = {"expires_at": 1000.0}
        assert _needs_refresh(tokens, now=939.0, leeway=60) is False

    def test_already_expired_needs_refresh(self):
        tokens = {"expires_at": 1000.0}
        assert _needs_refresh(tokens, now=1001.0, leeway=60) is True


class TestGetValidAccessToken:
    def test_raises_a_clear_error_when_no_tokens_are_stored(self, tmp_path):
        path = str(tmp_path / "tokens.json")
        with pytest.raises(OAuthError, match="oauth_login.py"):
            get_valid_access_token("client-id", path=path)

    def test_returns_existing_token_without_a_network_call_when_not_expired(
        self, tmp_path, monkeypatch
    ):
        path = str(tmp_path / "tokens.json")
        save_tokens(
            {
                "access_token": "still-good",
                "refresh_token": "r",
                "expires_at": time.time() + 3600,
            },
            path=path,
        )

        def _explode(*a, **k):
            raise AssertionError("urlopen must not be called when the token isn't expiring")

        monkeypatch.setattr(oauth.urllib.request, "urlopen", _explode)

        token = get_valid_access_token("client-id", path=path)

        assert token == "still-good"

    def test_refreshes_and_persists_when_within_the_leeway_window(self, tmp_path, monkeypatch):
        path = str(tmp_path / "tokens.json")
        save_tokens(
            {
                "access_token": "about-to-expire",
                "refresh_token": "old-refresh",
                "expires_at": time.time() + 10,  # inside the 60s leeway
            },
            path=path,
        )

        captured_requests = []

        def _fake_urlopen(req, timeout=None):
            captured_requests.append(req)
            return _FakeHTTPResponse(
                {"access_token": "refreshed", "refresh_token": "new-refresh", "expires_in": 3600}
            )

        monkeypatch.setattr(oauth.urllib.request, "urlopen", _fake_urlopen)

        token = get_valid_access_token("client-id", path=path)

        assert token == "refreshed"
        assert len(captured_requests) == 1
        assert captured_requests[0].full_url == oauth.TOKEN_ENDPOINT

        persisted = load_tokens(path=path)
        assert persisted["access_token"] == "refreshed"
        assert persisted["refresh_token"] == "new-refresh"

    def test_refresh_keeps_old_refresh_token_if_server_omits_a_new_one(
        self, tmp_path, monkeypatch
    ):
        path = str(tmp_path / "tokens.json")
        save_tokens(
            {
                "access_token": "about-to-expire",
                "refresh_token": "keep-me",
                "expires_at": time.time() + 10,
            },
            path=path,
        )

        def _fake_urlopen(req, timeout=None):
            # No refresh_token in this response, on purpose.
            return _FakeHTTPResponse({"access_token": "refreshed", "expires_in": 3600})

        monkeypatch.setattr(oauth.urllib.request, "urlopen", _fake_urlopen)

        get_valid_access_token("client-id", path=path)

        persisted = load_tokens(path=path)
        assert persisted["refresh_token"] == "keep-me"
