"""
oauth.py
========
OAuth 2.1 authorization-code + PKCE client for Robinhood's Agentic Trading
MCP server.

Server config verified 2026-07-30 against agent.robinhood.com's
.well-known discovery documents:
  - authorization_endpoint: https://robinhood.com/oauth
  - token_endpoint:         https://api.robinhood.com/oauth2/token/
  - PKCE required:          code_challenge_methods_supported == ["S256"]
  - public client:          token_endpoint_auth_method == "none"
  - grant_types:            authorization_code, refresh_token

This exists because a static bearer token cannot authenticate against this
server (README "Known Limitations" #11) — the server only accepts
short-lived access tokens minted through this flow and refreshed as
needed. Token values are never logged or printed anywhere in this module.
"""

import base64
import hashlib
import http.server
import json
import os
import secrets
import stat
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser

AUTHORIZATION_ENDPOINT = "https://robinhood.com/oauth"
TOKEN_ENDPOINT = "https://api.robinhood.com/oauth2/token/"
REDIRECT_PORT = 8765
REDIRECT_URI = f"http://localhost:{REDIRECT_PORT}/callback"
SCOPE = "internal"

# Outside the repo on purpose — this file holds live brokerage-account
# access/refresh tokens and must never end up in a git working tree.
DEFAULT_TOKENS_PATH = os.path.expanduser(os.path.join("~", ".robinhood-agent", "tokens.json"))

# Refresh proactively rather than waiting for a call to fail mid-cycle —
# a token good for only a few more seconds is treated as already expired.
REFRESH_LEEWAY_SECONDS = 60

# RFC 7636 requires a verifier of 43-128 characters. 96 random bytes
# base64url-encode to exactly 128 characters with no padding, landing at
# the top of that range.
_VERIFIER_BYTES = 96


class OAuthError(Exception):
    """Base class for OAuth flow failures."""


class OAuthStateMismatchError(OAuthError):
    """The authorization callback's state parameter didn't match what was
    sent. Treated as a possible CSRF attempt and rejected outright — this
    check is not optional and never skipped."""


def generate_pkce() -> tuple[str, str]:
    """Return (code_verifier, code_challenge) per RFC 7636 (S256 method).

    code_verifier: cryptographically random, URL-safe, 43-128 characters.
    code_challenge: base64url(sha256(code_verifier)), padding stripped.
    """
    verifier = secrets.token_urlsafe(_VERIFIER_BYTES)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return verifier, challenge


def _build_authorization_url(client_id: str, challenge: str, state: str) -> str:
    query = urllib.parse.urlencode(
        {
            "client_id": client_id,
            "redirect_uri": REDIRECT_URI,
            "response_type": "code",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "scope": SCOPE,
            "state": state,
        }
    )
    return f"{AUTHORIZATION_ENDPOINT}?{query}"


def _check_callback(params: dict, expected_state: str) -> str:
    """Validate the authorization callback's query params and return the
    authorization code, or raise.

    State is checked FIRST, unconditionally, before anything else about
    the response is trusted — this is the CSRF check and skipping it for
    any other field would defeat the point.
    """
    if params.get("state") != expected_state:
        raise OAuthStateMismatchError(
            "OAuth callback state did not match the value sent with the "
            "authorization request — rejecting to guard against CSRF."
        )
    if "error" in params:
        raise OAuthError(
            f"authorization server returned an error: {params.get('error')} "
            f"({params.get('error_description', 'no description')})"
        )
    code = params.get("code")
    if not code:
        raise OAuthError("authorization callback had no 'code' parameter")
    return code


class _CallbackCapture:
    """Holds whatever query params the one-shot local callback server
    receives, so the handler class (instantiated by http.server itself,
    not by us) has somewhere to put them."""

    def __init__(self):
        self.params: dict[str, str] = {}


def _make_callback_handler(capture: "_CallbackCapture"):
    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            capture.params = {
                k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()
            }
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body>Login complete. You can close this tab.</body></html>"
            )

        def log_message(self, format, *args):
            # Default BaseHTTPRequestHandler logging writes the full
            # request line — including the auth code in the query string
            # — to stderr. Silenced deliberately.
            pass

    return _Handler


def _token_request(form_data: dict) -> dict:
    body = urllib.parse.urlencode(form_data).encode("ascii")
    req = urllib.request.Request(
        TOKEN_ENDPOINT,
        data=body,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise OAuthError(f"token endpoint returned {e.code}: {detail}") from e
    return _normalize_token_response(raw)


def _normalize_token_response(raw: dict) -> dict:
    if "access_token" not in raw:
        raise OAuthError(f"token endpoint response had no access_token field: {raw}")
    return {
        "access_token": raw["access_token"],
        "refresh_token": raw.get("refresh_token"),
        "token_type": raw.get("token_type", "Bearer"),
        "expires_at": time.time() + float(raw.get("expires_in", 0)),
    }


def _exchange_code_for_tokens(client_id: str, code: str, verifier: str) -> dict:
    return _token_request(
        {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT_URI,
            "client_id": client_id,
            "code_verifier": verifier,
        }
    )


def _refresh_tokens(client_id: str, refresh_token: str) -> dict:
    tokens = _token_request(
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
        }
    )
    # Some authorization servers omit refresh_token from a refresh
    # response, meaning "keep using the one you sent" rather than "you no
    # longer have one" — carry the old value forward in that case.
    if not tokens.get("refresh_token"):
        tokens["refresh_token"] = refresh_token
    return tokens


def run_authorization_flow(client_id: str) -> dict:
    """Run the full interactive authorization-code + PKCE flow:

    1. Generate a PKCE verifier/challenge and a random CSRF state.
    2. Open the system browser to the authorization endpoint.
    3. Block on a single-request local HTTP server for the redirect.
    4. Verify state (reject on mismatch), extract the code.
    5. Exchange the code for tokens at the token endpoint.
    6. Persist the tokens and return them.
    """
    verifier, challenge = generate_pkce()
    state = secrets.token_urlsafe(32)
    auth_url = _build_authorization_url(client_id, challenge, state)

    capture = _CallbackCapture()
    server = http.server.HTTPServer(
        ("localhost", REDIRECT_PORT), _make_callback_handler(capture)
    )
    try:
        print(f"Opening your browser to authorize RobinhoodAgenticTrader...\n{auth_url}")
        webbrowser.open(auth_url)
        server.handle_request()  # blocks for exactly one request, then returns
    finally:
        server.server_close()

    code = _check_callback(capture.params, expected_state=state)
    tokens = _exchange_code_for_tokens(client_id, code, verifier)
    save_tokens(tokens)
    return tokens


def load_tokens(path: str = DEFAULT_TOKENS_PATH) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_tokens(tokens: dict, path: str = DEFAULT_TOKENS_PATH) -> None:
    """Write tokens to `path`, creating its parent directory if needed.

    Permissions are set to owner-only (0700 dir / 0600 file) via os.chmod.
    On POSIX this is enforced by the filesystem; on Windows os.chmod can
    only toggle the read-only attribute, so it's best-effort there — no
    stronger guarantee is available without a Windows-specific ACL call,
    which this project doesn't take on as a dependency.
    """
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    os.chmod(directory, stat.S_IRWXU)

    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(tokens, f)
    finally:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)


def _needs_refresh(tokens: dict, now: float, leeway: float = REFRESH_LEEWAY_SECONDS) -> bool:
    return now >= tokens["expires_at"] - leeway


def get_valid_access_token(
    client_id: str, path: str = DEFAULT_TOKENS_PATH, force_refresh: bool = False
) -> str:
    """Return a currently-valid access token, refreshing first if it's
    expired, expires within REFRESH_LEEWAY_SECONDS, or `force_refresh` is
    True. Raises OAuthError with an actionable message if no tokens have
    ever been stored.

    force_refresh exists for callers that got a 401 despite a locally
    unexpired token — that means the local expiry cache is wrong (e.g.
    server-side revocation, clock skew), and the ordinary expiry check
    would just hand back the same bad token again. See
    mcp_client._call_tool()'s one-retry-on-401 logic.
    """
    tokens = load_tokens(path)
    if tokens is None:
        raise OAuthError(
            "No stored Robinhood OAuth tokens found. Run "
            "`python scripts/oauth_login.py` to log in first."
        )

    if force_refresh or _needs_refresh(tokens, time.time()):
        tokens = _refresh_tokens(client_id, tokens["refresh_token"])
        save_tokens(tokens, path)

    return tokens["access_token"]
