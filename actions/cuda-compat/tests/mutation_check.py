#!/usr/bin/env python3
"""Mutation check: prove the suite fails when the evaluator is wrong.

A green suite only means something if a broken evaluator turns it red. This
applies each defect below to a copy of cuda-compat.py, runs the offline suite
against that copy, and requires the suite to FAIL. A mutation that survives is
reported as a hole in the tests, and this script exits non-zero.

Every mutation is a defect that would ship a wrong answer to a user rather than
an error, which is the only kind worth seeding. The first two are the highest
consequence available: inverting the OR/AND separators leaves the evaluator
running normally while making it wrong about every image with a driver-branch
allowance in its requirement string.

  ./mutation_check.py            run them all
  ./mutation_check.py --list     show the catalogue
"""
import argparse, os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(os.path.dirname(HERE), "cuda-compat.py")

# (name, why it matters, needle, replacement)
MUTATIONS = [
    ("or-and-inverted",
     "space=AND, comma=OR — every forward-compat allowance becomes ignorable noise",
     'terms = [t for t in expr.split(" ") if t]',
     'terms = [t for t in expr.split(",") if t]'),

    ("term-factors-inverted",
     "the other half of that inversion: factors split on space",
     'factors = [f for f in term.split(",") if f]',
     'factors = [f for f in term.split(" ") if f]'),

    ("undecided-assumed-true",
     "a missing fact treated as satisfied — the fail-open this tool exists to prevent",
     'return None, f"{factor} -> undecided (no {key} fact)", key',
     'return True, f"{factor} -> undecided (no {key} fact)", key'),

    ("cannot-verify-becomes-compatible",
     "CANNOT-VERIFY silently downgraded to a pass",
     "worst = {SATISFIED: 0, UNSATISFIED: 1, UNDECIDED: 2, MALFORMED: 1}[status]",
     "worst = {SATISFIED: 0, UNSATISFIED: 1, UNDECIDED: 0, MALFORMED: 1}[status]"),

    ("brand-ambiguity-ignored",
     "a verdict that depends on the GPU brand reported as if it were settled",
     "if len(unanimous) > 1:",
     "if False:"),

    ("untargeted-platform-votes",
     "an arm64 child deciding the verdict for an amd64 node",
     'targeted = p["arch"] is None or p["arch"] == target.arch',
     'targeted = True'),

    ("only-require-cuda-read",
     "the other NVIDIA_REQUIRE_* vars dropped, though the runtime ANDs them",
     'if k.startswith("NVIDIA_REQUIRE_") and not k.startswith("NVIDIA_REQUIRE_JETPACK") and v',
     'if k == "NVIDIA_REQUIRE_CUDA" and v'),

    # NB: mutating `n1 < n2` to `n1 <= n2` here would be an EQUIVALENT mutant —
    # it sits inside an `if n1 != n2` guard, so both spellings compute the same
    # function and no test could ever catch it. These two hit the tail rule,
    # where the comparator actually has room to be wrong.
    ("version-tail-zeros-not-skipped",
     "550.144.03.0 stops comparing equal to 550.144.03, unlike dsl_compare_version",
     'while i < len(v1) and v1[i] in ".0":',
     'while i < len(v1) and v1[i] in ".":'),

    ("version-tail-comparison-inverted",
     "the longer version reads as the smaller one once the shared components match",
     'if op in ("<", "<="):\n        return e1 and not e2',
     'if op in ("<", "<="):\n        return not e1 and e2'),

    ("brand-compare-case-sensitive",
     "brand=tesla no longer matching an NVML 'Tesla', unlike dsl_compare_string",
     "return s1.lower() == s2.lower()",
     "return s1 == s2"),

    ("legacy-cuda-version-ignored",
     "legacy images escaping the cuda>=X.Y the runtime synthesises for them",
     'if not env.get("NVIDIA_REQUIRE_CUDA") and env.get("CUDA_VERSION"):',
     "if False:"),

    ("arch-suffix-misread",
     "sm_120a read as an unmatched arch, so a correctly built Blackwell image looks uncompiled",
     'if base and base[-1] in ("a", "f"):',
     "if False:"),

    ("verdict-aggregation-by-exit-code",
     "CANNOT-VERIFY masks INCOMPATIBLE, and the gate goes green over a refused container",
     "    return max(codes, key=lambda c: SEVERITY[c]) if codes else default",
     "    return max(codes) if codes else default"),

    ("worse-picks-the-larger-exit-code",
     "the pairwise combine loses the same way as the reduction",
     "    return a if SEVERITY[a] >= SEVERITY[b] else b",
     "    return max(a, b)"),

    ("separator-term-vacuously-satisfied",
     "a term of nothing but separators reads as satisfied — false COMPATIBLE on a comma",
     "        if not factors:",
     "        if False:"),

    ("legacy-synthesis-gated-on-all-require-vars",
     "any NVIDIA_REQUIRE_* suppresses the runtime's synthesized cuda>= — a dropped conjunct",
     '    if not env.get("NVIDIA_REQUIRE_CUDA") and env.get("CUDA_VERSION"):',
     '    if not reqs and env.get("CUDA_VERSION"):'),

    ("ptx-direction-ignored",
     "PTX from a NEWER arch treated as usable on an older GPU",
     "    if ptx and any(vtuple(p) <= vtuple(cc) for p in ptx):",
     "    if ptx:"),

    ("first-index-child-only",
     "the single-request regression: only the first child manifest is read",
     'for m in top["manifests"]:',
     'for m in top["manifests"][:1]:'),

    ("unknown-driver-branch-guessed",
     "an unknown driver branch quietly evaluated instead of refused",
     "if not target.max_cuda:",
     "if False:"),
]


def run_suite(script_path):
    """-> (passed, tail of output). The suite is pointed at a mutant via MUTANT_PATH."""
    env = dict(os.environ, MUTANT_PATH=script_path)
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "test_cuda_compat", "test_dsl_parity", "-v"],
        cwd=HERE, env=env, capture_output=True, text=True)
    return proc.returncode == 0, (proc.stderr or proc.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for name, why, _, _ in MUTATIONS:
            print(f"  {name:<34} {why}")
        return 0

    with open(SCRIPT) as f:
        original = f.read()

    print("==> baseline: the unmutated suite must pass")
    ok, out = run_suite(SCRIPT)
    if not ok:
        print(out[-3000:])
        print("BASELINE FAILED — fix the suite before trusting any mutation result.")
        return 1
    print("    baseline green\n")

    survivors, applied = [], 0
    tmpdir = tempfile.mkdtemp(prefix="cuda-compat-mutants-")
    try:
        for name, why, needle, replacement in MUTATIONS:
            if needle not in original:
                print(f"[STALE ] {name}: needle no longer present in cuda-compat.py")
                survivors.append((name, "stale mutation — the code it targeted moved"))
                continue
            if original.count(needle) != 1:
                print(f"[STALE ] {name}: needle matches {original.count(needle)} places, need exactly 1")
                survivors.append((name, "ambiguous mutation needle"))
                continue
            mutant_path = os.path.join(tmpdir, f"{name}.py")
            with open(mutant_path, "w") as f:
                f.write(original.replace(needle, replacement))
            applied += 1
            ok, out = run_suite(mutant_path)
            if ok:
                print(f"[SURVIVED] {name} — {why}")
                survivors.append((name, why))
            else:
                failed = [ln for ln in out.splitlines() if ln.startswith(("FAIL:", "ERROR:"))]
                print(f"[caught ] {name:<34} {len(failed)} test(s) red")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"\n{applied - len(survivors)}/{applied} mutations caught")
    if survivors:
        print("\nSURVIVING MUTATIONS — the suite cannot tell these defects from correct code:")
        for name, why in survivors:
            print(f"  - {name}: {why}")
        return 1
    print("every seeded defect turns the suite red")
    return 0


if __name__ == "__main__":
    sys.exit(main())
