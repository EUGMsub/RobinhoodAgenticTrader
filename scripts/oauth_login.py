#!/usr/bin/env python3
"""
oauth_login.py
===============
One-time interactive login: runs the OAuth 2.1 authorization-code + PKCE
flow against Robinhood's Agentic Trading MCP server and stores the
resulting access/refresh tokens locally (see oauth.DEFAULT_TOKENS_PATH).

Run this once before running the live agent, and again any time the token
file is deleted or the server revokes the refresh token.

Usage:
    python scripts/oauth_login.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import AgentConfig
from oauth import DEFAULT_TOKENS_PATH, OAuthError, run_authorization_flow


def main() -> None:
    cfg = AgentConfig()
    if not cfg.robinhood_client_id:
        sys.exit(
            "Missing required environment variable: ROBINHOOD_CLIENT_ID. "
            "See .env.example."
        )

    print("Starting Robinhood OAuth login...")
    try:
        run_authorization_flow(cfg.robinhood_client_id)
    except OAuthError as e:
        sys.exit(f"Login failed: {e}")

    print(f"Login succeeded. Tokens stored at {DEFAULT_TOKENS_PATH}.")


if __name__ == "__main__":
    main()
