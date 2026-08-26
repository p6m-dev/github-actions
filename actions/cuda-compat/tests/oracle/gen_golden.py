#!/usr/bin/env python3
"""Build the golden parity corpus by asking the compiled C oracle for verdicts.

Invoked by refresh.sh, which owns the upstream pin and the sha256 check.

The corpus is deliberately adversarial toward cuda-compat.py's port: the real
image expressions (where a wrong AND/OR reading flips the answer), the driver
branches around each enumerated range, brands inside and outside the enumerated
set, version-tail edge cases the C comparator handles unusually (leading zeros,
uneven component counts, all-zero tails), and expressions the C code rejects
outright (unknown property, missing operator, ordering comparison on a string).
"""
import itertools, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(os.path.dirname(HERE), "fixtures")

# (driver, cuda, arch, brand) — facts a target can state.
FACTS = [
    ("550.144.03", "12.4", "7.5", "Tesla"),     # the T4 pools as declared today
    ("550.144.03", "12.4", "7.5", "Nvidia"),    # same node, brand as newer NVML reports it
    ("550.144.03", "12.4", "7.5", "GeForce"),   # consumer card: forward compat is not offered
    ("580.65.06", "13.0", "12.0", "Nvidia"),    # the flex node
    ("545.23.08", "12.3", "7.5", "Tesla"),      # a branch between enumerated ones
    ("535.183.06", "12.2", "7.5", "Tesla"),
    ("470.256.02", "11.4", "7.5", "Tesla"),
    ("570.86.15", "12.8", "8.6", "Nvidia"),
]

# Expressions that exercise the comparator and the parser rather than real images.
SYNTHETIC = [
    "cuda>=12.4",
    "cuda>12.4",
    "cuda=12.4",
    "cuda!=12.4",
    "cuda<=12.4",
    "cuda<12.4",
    "driver>=550",
    "driver<551",
    "driver>=550.144.03",
    "driver>=550.144.3",          # leading zero vs not: C compares numerically
    "driver>=550.144.04",
    "driver<=550.144.03.0",       # all-zero tail counts as absent
    "driver=550.144.03",
    "driver>=550.144",            # uneven component counts
    "driver<550.145",
    # The tail rule, where the comparator decides between versions whose shared
    # components all match. An expression that differs earlier (550.145) exits
    # before reaching it, so these are the only cases that exercise it.
    "driver>=550.144.03.0",       # all-zero tail: equal, so >= holds
    "driver=550.144.03.0",
    "driver<550.144",             # 550.144.03 is the LONGER, hence larger, version
    "driver<=550.144",
    "driver>550.144",
    # A fact whose own tail is a zero component (cuda 13.0) against a shorter
    # value: the only shape where the fact-side zero-skip changes the answer.
    "cuda=13",
    "cuda<=13",
    "cuda>13",
    "cuda!=13",
    "arch>=7.5",
    "arch>7.5",
    "arch=7.5",
    "brand=tesla",
    "brand=TESLA",                # comparison is case-insensitive
    "brand!=geforce",
    "cuda>=13.0 driver>=550,driver<551",             # OR of a cuda term and a range term
    "cuda>=13.0,driver>=550",                        # AND: both must hold
    "brand=tesla,driver>=550,driver<551",
    "cuda>=99.0 brand=tesla,driver>=550,driver<551",
    "cuda>=99.0 brand=nosuchbrand,driver>=550",      # no term can pass
    "cuda>=13.0  driver>=550,driver<551",            # double space: empty term skipped
    # Separator-only terms. The inner loop consumes nothing, which C cannot tell
    # apart from "every factor passed", so the OR scan STOPS there carrying
    # whatever the previous term decided — and everything after it is unreachable.
    "cuda>=13.0 , brand=tesla,driver>=550,driver<551",
    "cuda>=13.0 ,",
    ", cuda>=99.0",
    ",",
    "brand=tesla ,",
    "cuda>=99.0 ,, brand=tesla,driver>=550,driver<551",
    "brand<tesla",                                   # ordering on a string: invalid
    "cuda>=13.0 brand<tesla",                        # invalid, but only if reached
    "nosuchkey>=1",                                  # unknown property: invalid
    "cuda>=13.0 nosuchkey>=1",                       # unknown property after a passing term
    "cuda",                                          # no operator: invalid
    "cuda>=",                                        # no value: invalid
]


def image_expressions():
    out = []
    for name in sorted(os.listdir(FIXTURES)):
        if not name.endswith(".json"):
            continue
        fx = json.load(open(os.path.join(FIXTURES, name)))
        for p in fx["platforms"]:
            for k, v in p["env"].items():
                if k.startswith("NVIDIA_REQUIRE_") and v:
                    out.append((f"{name[:-5]}:{p['platform']}:{k}", v))
    return out


def ask(oracle, driver, cuda, arch, brand, expr):
    r = subprocess.run([oracle, driver, cuda, arch, brand, expr],
                       capture_output=True, text=True)
    out = r.stdout.strip()
    if out.startswith("SATISFIED"):
        return "SATISFIED", None
    reason = out.split("UNSATISFIED: ", 1)[-1]
    # dsl_evaluate reports "invalid expression" for a term it cannot parse or a
    # property it does not know; everything else is an honest unsatisfied verdict.
    return ("MALFORMED" if reason.strip() == "invalid expression" else "UNSATISFIED"), reason


def main():
    oracle, out_path = sys.argv[1], sys.argv[2]
    exprs = [("synthetic", e) for e in SYNTHETIC] + image_expressions()
    cases = []
    for (source, expr), (driver, cuda, arch, brand) in itertools.product(exprs, FACTS):
        status, reason = ask(oracle, driver, cuda, arch, brand, expr)
        cases.append({"source": source, "expr": expr, "driver": driver, "cuda": cuda,
                      "arch": arch, "brand": brand, "expect": status, "reason": reason})
    doc = {
        "note": ("Verdicts produced by compiling libnvidia-container's own src/cli/dsl.c "
                 "verbatim with the four rules src/cli/configure.c registers. Regenerate "
                 "with refresh.sh only when the upstream pin moves deliberately."),
        "upstream_repo": "https://github.com/NVIDIA/libnvidia-container",
        "upstream_commit": os.environ.get("UPSTREAM_COMMIT", "unknown"),
        "dsl_c_sha256": os.environ.get("DSL_SHA256", "unknown"),
        "cases": cases,
    }
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    counts = {}
    for c in cases:
        counts[c["expect"]] = counts.get(c["expect"], 0) + 1
    print(f"wrote {out_path}: {len(cases)} cases {counts}")


if __name__ == "__main__":
    main()
