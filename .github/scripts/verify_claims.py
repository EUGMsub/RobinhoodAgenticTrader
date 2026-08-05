"""
verify_claims.py
=================
CI-only. Automates two checks the pre-commit-review skill otherwise does by
hand (see .claude/skills/pre-commit-review):

1. The test count README.md claims ("N tests (N pass, N skipped by
   default)") matches what `pytest tests/ -q` actually reports. README
   drift on this exact claim has happened before in this project.
2. No test file imports a live network client (anthropic, requests, httpx)
   outside tests/test_injection_live.py, the one module deliberately gated
   behind RUN_LIVE_INJECTION_TESTS — the README claims the rest of the
   suite makes no network calls; this verifies it instead of trusting it.

Exits non-zero with a specific message on any mismatch or violation. Not a
test itself (pytest never collects this file) — it's a standalone script
run as its own CI step.
"""

import ast
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
README_PATH = REPO_ROOT / "README.md"
TESTS_DIR = REPO_ROOT / "tests"

# Modules whose mere presence in a test file means that file can make a
# real network call to a real service. urllib is deliberately NOT here:
# it's stdlib, mcp_client.py itself is built on it, and every test file
# that references it (test_mcp_client.py, test_oauth.py) only ever
# monkeypatches urlopen — it never calls it for real.
LIVE_NETWORK_MODULES = {"anthropic", "requests", "httpx"}
GATED_LIVE_TEST_MODULE = "test_injection_live.py"

# Matches this project's README wording exactly: "265 tests (256 pass, 9
# skipped by default)". If the README's phrasing ever changes, this needs
# to change with it — that's a feature, not a bug: a silently-passing
# check that stopped checking anything would be exactly the kind of
# "absence rendered as presence" bug this project explicitly watches for.
CLAIM_RE = re.compile(r"(\d+)\s+tests\s*\((\d+)\s+pass,\s*(\d+)\s+skipped by default\)")

# Matches pytest -q's final summary line, e.g. "256 passed, 9 skipped in
# 3.91s" or "3 failed, 253 passed, 9 skipped in 4.02s".
SUMMARY_RE = re.compile(
    r"^(?:(?P<failed>\d+) failed, )?(?P<passed>\d+) passed"
    r"(?:, (?P<skipped>\d+) skipped)? in [\d.]+s",
    re.MULTILINE,
)


def check_test_count() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--color=no"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    match = SUMMARY_RE.search(output)
    if not match:
        print("verify_claims: could not parse a pytest summary line out of:")
        print(output[-4000:])
        sys.exit(1)

    failed = int(match.group("failed") or 0)
    passed = int(match.group("passed") or 0)
    skipped = int(match.group("skipped") or 0)

    if failed:
        print(f"verify_claims: {failed} test(s) FAILED — fix the suite before verifying claims about it.")
        print(output[-4000:])
        sys.exit(1)

    actual_total = failed + passed + skipped

    readme_text = README_PATH.read_text(encoding="utf-8")
    claims = sorted(set(CLAIM_RE.findall(readme_text)))
    if not claims:
        print(
            "verify_claims: README.md doesn't contain a test count in the "
            "expected 'N tests (N pass, N skipped by default)' shape — "
            "nothing to check it against. If the wording changed "
            "intentionally, update CLAIM_RE in this script to match."
        )
        sys.exit(1)
    if len(claims) > 1:
        print(f"verify_claims: README.md states {len(claims)} DIFFERENT test-count claims: {claims}")
        sys.exit(1)

    claimed_total, claimed_pass, claimed_skip = (int(x) for x in claims[0])

    if (claimed_total, claimed_pass, claimed_skip) != (actual_total, passed, skipped):
        print(
            "verify_claims: README.md's test-count claim does not match reality.\n"
            f"  README claims:  {claimed_total} tests ({claimed_pass} pass, {claimed_skip} skipped)\n"
            f"  pytest reports: {actual_total} tests ({passed} pass, {skipped} skipped)\n"
            "Update README.md (or fix the test suite) so they agree."
        )
        sys.exit(1)

    print(
        f"verify_claims: README test-count claim matches pytest — "
        f"{actual_total} tests ({passed} pass, {skipped} skipped)."
    )


def check_no_live_network_imports() -> None:
    violations = []
    for test_file in sorted(TESTS_DIR.glob("*.py")):
        if test_file.name == GATED_LIVE_TEST_MODULE:
            continue
        tree = ast.parse(test_file.read_text(encoding="utf-8"), filename=str(test_file))
        hits: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                hits |= {alias.name.split(".")[0] for alias in node.names} & LIVE_NETWORK_MODULES
            elif isinstance(node, ast.ImportFrom) and node.module:
                hits |= {node.module.split(".")[0]} & LIVE_NETWORK_MODULES
        if hits:
            violations.append(f"{test_file.relative_to(REPO_ROOT)}: imports {', '.join(sorted(hits))}")

    if violations:
        print(
            "verify_claims: README claims the suite makes no network calls "
            f"outside {GATED_LIVE_TEST_MODULE}, but these files import a live "
            "network client:"
        )
        for v in violations:
            print(f"  {v}")
        sys.exit(1)

    print(f"verify_claims: no live network client imports outside {GATED_LIVE_TEST_MODULE}.")


if __name__ == "__main__":
    check_test_count()
    check_no_live_network_imports()
    print("verify_claims: all claims verified.")
