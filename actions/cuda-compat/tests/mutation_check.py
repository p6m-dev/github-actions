#!/usr/bin/env python3
"""Mutation check: prove the suite fails when the code is wrong.

A green suite only means something if broken code turns it red. This applies each
defect below to a copy of the script it targets, runs that script's suite against
the copy, and requires the suite to FAIL. A mutation that survives is reported as
a hole in the tests, and this script exits non-zero.

Two targets, because there are two ways to be confidently wrong:

  evaluator (cuda-compat.py)     — judging an image against a target incorrectly.
                                   The first two mutations are the highest
                                   consequence available: inverting the OR/AND
                                   separators leaves the evaluator running
                                   normally while making it wrong about every
                                   image with a driver-branch allowance.
  resolver  (resolve-targets.py) — judging it against the WRONG TARGET. A fully
                                   correct verdict about a machine that does not
                                   exist reads exactly like a correct one.

Every mutation is a defect that would ship a wrong answer to a user rather than
an error, which is the only kind worth seeding.

  ./mutation_check.py            run them all
  ./mutation_check.py --list     show the catalogue
"""
import argparse, os, shutil, subprocess, sys, tempfile

try:
    import yaml  # noqa: F401  (availability probe, not a use)
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False

HERE = os.path.dirname(os.path.abspath(__file__))
ACTION = os.path.dirname(HERE)
SCRIPT = os.path.join(ACTION, "cuda-compat.py")
RESOLVER = os.path.join(ACTION, "resolve-targets.py")

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

# Defects in how a pool's target is DERIVED. These matter as much as the
# evaluator's: a target class names the driver an image is judged against, so a
# wrong one produces a fully-reasoned verdict about a machine that does not
# exist. Deriving the driver from the karpenter-azure pin was exactly this
# defect, shipped — hence the first two.
RESOLVER_MUTATIONS = [
    ("per-pool-override-ignored",
     "every pool judged at the cluster default, so a pool running another driver "
     "is verdicted against one it does not have",
     'return {pool: str(overrides.get(pool, base)) for pool in pools}, None',
     'return {pool: str(base) for pool in pools}, None'),

    ("driver-major-accepted-as-a-version",
     "\"580\" accepted as a driver fact, though it installs nothing and names no image",
     r'DRIVER_VERSION = re.compile(r"^\d{3,4}\.\d+\.\d+$")',
     r'DRIVER_VERSION = re.compile(r"^\d{3,4}")'),

    ("chart-default-sku-reintroduced",
     "the chart's default copied back in — the drift that silently moved every "
     "inheriting pool's compute capability when it changed upstream",
     '        sku_names = gpu.get("skuGpuNames")',
     '        sku_names = gpu.get("skuGpuNames") or ["A10"]'),

    ("grid-model-modelled-anyway",
     "a pool invented for a GPU whose render fails, judged against a driver it "
     "could never receive",
     '            grid = [s for s in sku_names if gpus[s].get("driver_family") == "grid"]',
     '            grid = []'),

    ("cloud-assumed-azure",
     "an AWS pool read as Azure, so the AMI's driver is replaced by a declaration "
     "that does nothing there",
     '        azure = has_addon(config, configs, "karpenter-azure")',
     '        azure = True'),

    ("impossible-silicon-pair-allowed",
     "a driver older than the GPU blessed, though no such driver has seen that silicon",
     '                if int(branch) < int(spec["min_driver_branch"]):',
     '                if False:'),

    ("stranded-pool-not-refused",
     "a declared version no NVIDIADriver CR installs passed as checked — the pool "
     "whose nodes never receive a driver at all",
     '            if crs is not None and azure and driver not in crs:',
     '            if False:'),
]


def run_suite(env_var, script_path, modules):
    """-> (passed, tail of output). The suite is pointed at a mutant via env_var."""
    env = dict(os.environ, **{env_var: script_path})
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", *modules, "-v"],
        cwd=HERE, env=env, capture_output=True, text=True)
    return proc.returncode == 0, (proc.stderr or proc.stdout)


# (label, script, env var, test modules, mutations, where mutants may be written)
#
# Resolver mutants must sit beside the original: resolve-targets.py locates
# tables/ relative to its own path, so a copy in a tmpdir would fail to import
# for a reason that has nothing to do with the seeded defect — and a mutation
# "caught" by its own plumbing proves nothing.
TARGETS = [
    ("evaluator", SCRIPT, "MUTANT_PATH",
     ["test_cuda_compat", "test_dsl_parity"], MUTATIONS, None),
    ("resolver", RESOLVER, "RESOLVER_MUTANT_PATH",
     ["test_resolve_targets"], RESOLVER_MUTATIONS, ACTION),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for label, _, _, _, mutations, _ in TARGETS:
            print(f"  {label}:")
            for name, why, _, _ in mutations:
                print(f"    {name:<34} {why}")
        return 0

    survivors, applied = [], 0
    tmpdir = tempfile.mkdtemp(prefix="cuda-compat-mutants-")
    written = []
    try:
        for label, script, env_var, modules, mutations, mutant_dir in TARGETS:
            if label == "resolver" and not HAVE_YAML:
                # Every resolver test would skip, the suite would pass, and all
                # seven mutations would "survive" — or worse, be waved through as
                # a known-flaky. Neither reading is true, so refuse instead.
                print("==> resolver: PyYAML absent, so its tests would all skip and this "
                      "check would measure nothing. pip install pyyaml and re-run.")
                return 1
            print(f"==> {label}: baseline must pass")
            ok, out = run_suite(env_var, script, modules)
            if not ok:
                print(out[-3000:])
                print("BASELINE FAILED — fix the suite before trusting any mutation result.")
                return 1
            print("    baseline green")

            with open(script) as f:
                original = f.read()
            for name, why, needle, replacement in mutations:
                if needle not in original:
                    print(f"[STALE ] {name}: needle no longer present in {os.path.basename(script)}")
                    survivors.append((name, "stale mutation — the code it targeted moved"))
                    continue
                if original.count(needle) != 1:
                    print(f"[STALE ] {name}: needle matches {original.count(needle)} places, "
                          f"need exactly 1")
                    survivors.append((name, "ambiguous mutation needle"))
                    continue
                target_dir = mutant_dir or tmpdir
                mutant_path = os.path.join(target_dir, f".mutant-{name}.py")
                with open(mutant_path, "w") as f:
                    f.write(original.replace(needle, replacement))
                written.append(mutant_path)
                applied += 1
                ok, out = run_suite(env_var, mutant_path, modules)
                if ok:
                    print(f"[SURVIVED] {name} — {why}")
                    survivors.append((name, why))
                else:
                    failed = [ln for ln in out.splitlines() if ln.startswith(("FAIL:", "ERROR:"))]
                    print(f"[caught ] {name:<34} {len(failed)} test(s) red")
            print()
    finally:
        for path in written:
            try:
                os.unlink(path)
            except OSError:
                pass
        shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"{applied - len(survivors)}/{applied} mutations caught")
    if survivors:
        print("\nSURVIVING MUTATIONS — the suite cannot tell these defects from correct code:")
        for name, why in survivors:
            print(f"  - {name}: {why}")
        return 1
    print("every seeded defect turns the suite red")
    return 0


if __name__ == "__main__":
    sys.exit(main())
