# The differential oracle

`golden.json` holds verdicts produced by **libnvidia-container's own requirement
evaluator**, not by anything written here. `test_dsl_parity.py` replays every
case through `cuda-compat.py`'s Python port and fails if the two ever disagree.

## Why this exists

The evaluator's semantics are easy to state and easy to get backwards:

> space-separated constraints are ORed, comma-separated constraints are ANDed

Swap those two and nothing crashes. A clause like
`brand=tesla,driver>=550,driver<551` collapses into always-true noise, every
verdict quietly reduces to the leading `cuda>=13.0` term, and images that run
perfectly well are reported INCOMPATIBLE with a confident-looking explanation
attached. That clause is not noise: it is how an NVIDIA base image declares which
driver branches its bundled `cuda-compat` package supports.

A single inverted operator is therefore enough to make this tool wrong about
every image it looks at, in a way that reads as competence. Reviewing the
semantics by eye cannot rule that out, so the suite does not try: it executes
NVIDIA's code and demands agreement.

## How the table is produced

`refresh.sh`:

1. fetches `NVIDIA/libnvidia-container` at a **pinned commit** (never a branch),
2. verifies `src/cli/dsl.c` against a recorded sha256 and **refuses** if it moved
   — a changed evaluator is a review event, not a silent re-baseline,
3. compiles that file *verbatim* against `harness.c`, which registers the same
   four rules (`cuda`, `driver`, `arch`, `brand`) that `src/cli/configure.c`
   registers, with the same "no device visible → assume ok" bodies,
4. runs `gen_golden.py`, which asks the compiled oracle for a verdict on every
   corpus case and writes `golden.json` with the provenance attached.

Only `harness.c`, `cli.h` and `utils.h` live here — small stubs supplying the
handful of symbols `dsl.c` needs (`struct error`, `xstrdup`, `str_case_equal`,
`nitems`). `dsl.c` itself is never vendored; it is fetched at the pin.

## Running it

```bash
./refresh.sh          # needs network + a C compiler; run by hand, never in CI
```

CI does **not** run this. The whole point of a golden table is that the test
suite needs no compiler, no network, and no third-party source at test time.

## When a parity test fails

Do not re-run `refresh.sh` to make it green. A parity failure means
`cuda-compat.py` and NVIDIA disagree about what a real image does on a real node
— which is the exact bug class this tool exists to prevent, and the one that
looks like competence right up until a container refuses to start.

Re-baseline only when NVIDIA genuinely changed the evaluator. That path is
deliberately noisy: `refresh.sh` aborts on the sha256 mismatch and makes you read
the upstream diff, decide what it means for the port, and move `UPSTREAM_COMMIT`
and `DSL_SHA256` together in one reviewed commit.

## The corpus

`gen_golden.py` crosses every expression with every fact set. It is built to be
adversarial toward the port rather than flattering to it:

- **real image expressions** pulled straight from `../fixtures/*.json` — the
  strings where a wrong AND/OR reading flips the answer;
- **driver branches around each enumerated allowance** — 545 and 555 sit between
  supported branches and must be refused while 550 is accepted;
- **brands inside and outside the enumerated set** — `Tesla` accepted, `GeForce`
  refused on the same driver, because forward compatibility is offered to
  datacenter GPUs only;
- **version-tail edge cases** — leading zeros (`550.144.03`), uneven component
  counts, all-zero tails, which the C comparator handles in a way no
  general-purpose semver library reproduces;
- **expressions the runtime rejects outright** — unknown property, missing
  operator, ordering comparison on a string.

`test_dsl_parity.py` asserts the corpus keeps all three verdict classes with at
least 20 cases each, so it cannot decay into a table that would pass any port.
