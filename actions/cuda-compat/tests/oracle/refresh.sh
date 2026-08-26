#!/usr/bin/env bash
# Regenerate golden.json by executing libnvidia-container's OWN requirement
# evaluator. Run this by hand when the pin below moves — never in CI: the point
# of the golden table is that the test suite needs no compiler, no network, and
# no third-party source at test time.
#
# What it does:
#   1. fetches libnvidia-container at a pinned commit (never a moving ref)
#   2. verifies src/cli/dsl.c against the sha256 recorded here, and refuses if it
#      moved — a changed evaluator is a review event, not a silent re-baseline
#   3. compiles that file verbatim with the four rules src/cli/configure.c
#      registers (harness.c), and records its verdict for every corpus case
#
# Refuse the temptation to "fix" a failing parity test by re-running this.
# A parity failure means cuda-compat.py and NVIDIA disagree, which is the exact
# thing this suite exists to catch.
set -euo pipefail

UPSTREAM_REPO="https://github.com/NVIDIA/libnvidia-container.git"
UPSTREAM_COMMIT="00b4b47e2a3eed09f8059eb39feaa0b4540d22d9"  # 2026-08-24
DSL_SHA256="b5572974cc2e2f1a091caaa7c610ae2ac30e310c4b9e530fe247bd81a64b336f"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

echo "==> fetching libnvidia-container @ ${UPSTREAM_COMMIT:0:12}"
git -C "$work" init -q repo
git -C "$work/repo" remote add origin "$UPSTREAM_REPO"
git -C "$work/repo" fetch -q --depth 1 origin "$UPSTREAM_COMMIT"
git -C "$work/repo" checkout -q FETCH_HEAD

actual="$(shasum -a 256 "$work/repo/src/cli/dsl.c" | cut -d' ' -f1)"
if [ "$actual" != "$DSL_SHA256" ]; then
  echo "REFUSING: src/cli/dsl.c is $actual, expected $DSL_SHA256." >&2
  echo "NVIDIA changed the evaluator. Read the diff, decide what it means for" >&2
  echo "cuda-compat.py, then update UPSTREAM_COMMIT and DSL_SHA256 together." >&2
  exit 1
fi
echo "==> dsl.c matches the pinned sha256"

cp "$work/repo/src/cli/dsl.c" "$work/repo/src/cli/dsl.h" "$work/"
cp "$here/harness.c" "$here/cli.h" "$here/utils.h" "$work/"
cc -O0 -o "$work/oracle" "$work/harness.c" "$work/dsl.c" -I"$work"
echo "==> harness built"

UPSTREAM_COMMIT="$UPSTREAM_COMMIT" DSL_SHA256="$DSL_SHA256" \
  python3 "$here/gen_golden.py" "$work/oracle" "$here/golden.json"
