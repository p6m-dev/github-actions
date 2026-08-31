# cuda-compat

**Will this image run on that GPU pool?** Answered from image metadata alone —
no cluster, no GPU, no image pull, seconds per image.

It evaluates the image's `NVIDIA_REQUIRE_*` expressions exactly as
`nvidia-container-cli` does at container start, using a port of
libnvidia-container's own DSL evaluator that is
[differentially tested against NVIDIA's C code](tests/oracle/README.md).

```bash
python3 cuda-compat.py vllm/vllm-openai:v0.25.1 --target 550.144.03/7.5/amd64/Tesla
```

## Use it in a workflow

```yaml
- uses: p6m-dev/github-actions/actions/cuda-compat@<commit-sha>   # pin to a SHA
  with:
    images: ${{ steps.resolve.outputs.images }}
    targets: ${{ steps.resolve.outputs.targets }}
```

## The thing to understand before reading a verdict

`NVIDIA_REQUIRE_CUDA` is not a minimum-version field. It is a boolean expression
where **space is OR and comma is AND**. A real one looks like:

```
cuda>=13.0 brand=tesla,driver>=550,driver<551 brand=nvidia,driver>=550,driver<551 …
```

That reads: *CUDA 13.0 or newer, **or** a Tesla-brand driver in the 550 branch,
**or** …. Those branch clauses are how an NVIDIA base image declares which driver
branches its bundled `cuda-compat` package supports — CUDA Forward
Compatibility. So an image built against CUDA 13 genuinely runs on a 550 driver,
on a datacenter GPU, and the same image is refused on a GeForce.

Get the operators backwards and every one of those clauses collapses into
always-true noise, leaving only `cuda>=13.0` — turning runnable images into
confident INCOMPATIBLE verdicts, with no error to notice. Hence the oracle: the
suite does not reason about the semantics, it executes NVIDIA's code and demands
agreement.

## Verdicts

| verdict | exit | meaning |
|---|---|---|
| `COMPATIBLE` | 0 | the driver check will not refuse this container. The **basis** says whether that is a native CUDA-version match or reliance on forward compatibility — and names any axis that was *not* measured, such as kernel fit when the image declares no `TORCH_CUDA_ARCH_LIST`. |
| `INCOMPATIBLE` | 1 | it will be refused, or the image declares an arch list that does not cover the GPU. The exact failing constraint is quoted, including the message the operator will see in the log. Outranks CANNOT-VERIFY when a run mixes both. |
| `CANNOT-VERIFY` | 2 | a fact is missing, or the image declares no CUDA metadata. **Never a silent pass.** |

The exit code is always one of these three, so the signal is never lost.
Consumers pick their own strictness — the action's `strict-unknown` decides
whether exit 2 fails the step.

### Native vs. forward-compat compatibility

Both are `COMPATIBLE`, and the difference is operationally real. Forward compat
depends on the image's bundled `cuda-compat` libraries, applies to datacenter
GPUs only, and covers only the branches the image enumerates — so an image that
is COMPATIBLE at 550 today can become INCOMPATIBLE at 555 tomorrow, because 555
is simply not in its list. `basis` is there so a driver bump can be reasoned
about instead of hoped through.

## Facts, and refusing to invent them

A target is `[LABEL=]DRIVER[/CC[/ARCH[/BRAND,BRAND]]]`:

- **DRIVER** — full version (`550.144.03`) when known. A bare branch (`550`)
  cannot settle a comparison against a finer version, and says so.
- **CC** — GPU compute capability, NVIDIA's `arch` property.
- **ARCH** — which image variant runs here (default `amd64`). Other variants are
  reported but never vote: an arm64 child cannot run on an amd64 node.
- **BRAND** — NVML brand(s) the GPU may report. Datacenter SKUs report different
  values across driver branches, so a set is allowed and the verdict must hold
  for every member.

Missing facts make the terms that need them **undecided**, never assumed-true; a
verdict is issued only when it holds however the unknown resolves. Contradictory
facts are refused outright — `--from-node` will not verdict a node whose observed
driver disagrees with its declared label.

`resolve-targets.py` derives targets mechanically from `platform-versions`, so a
new pool is covered the moment its VersionConfig merges. One target class per
rendered **pool** per GPU model:

| cloud | where the driver comes from |
|---|---|
| Azure | `gpu.driverVersion`, narrowed per pool by `gpu.driverVersionOverrides` |
| AWS | `gpu.cudaDriverMajor` — branch only; the AMI decides the rest |

Azure GPU nodes are provisioned with **no baked driver** (`AKSNodeClass
spec.gpu.mode: None`) and the gpu-operator installs the version each pool
declares, so the pool's own declaration is the driver fact. Taking it from the
karpenter-azure pin instead names the provider's compiled-in constant, which no
node receives — a confident wrong answer at exit 0, and blind to per-pool
overrides.

The only hand-maintained rows left are GPU model → compute capability and NVML
brands, which nothing upstream publishes machine-readably. A model missing from
that table refuses the pool **by name**, never skips it.

`--driver-crs <path>` additionally refuses a pool whose declared version no
`NVIDIADriver` CR installs — the failure that strands a pool with GPU pods
pending forever. Declaring a version and installing it are two values in two
repos with nothing enforcing the pair.

## What it does not tell you

The assurance tier is **static analysis of image config**, not "boots":

- It cannot see whether the bundled compat libraries are intact, only that the
  image claims them.
- `TORCH_CUDA_ARCH_LIST` is checked against the compute capability, but that axis
  is **not enforced at container start** — a mismatch surfaces later as a kernel
  launch failure, and the finding says so.
- **Kernel fit is only checked when the image declares it.** Four of the five
  images in our own effective set declare no `TORCH_CUDA_ARCH_LIST` at all, so
  that axis is reported as unverified rather than guessed, and the verdict rests
  on the driver check alone. The reason column says so on every such row. A
  documented upstream floor that the image does not declare — SGLang's `sm_80+`,
  say — is invisible here by construction.
- **Arch-accelerated targets are only partly visible.** `sm_100a` / `sm_120a`
  carry Blackwell's `tcgen05` instructions and NVFP4/MXFP6 datatypes; the suffix
  is parsed, so a `12.0a` entry matches a 12.0 GPU, but an image listing plain
  `12.0` is indistinguishable from one built with the accelerated paths. A
  COMPATIBLE verdict on Blackwell means *it starts and kernels exist for that
  family*, not *the fast paths are compiled in*. The same caveat sharpens the PTX
  note: generic PTX JIT'd onto `sm_120` can never reach an arch-accelerated
  instruction, so "works via PTX" there means "runs without the hardware you paid
  for".
- A runtime configured for CDI, or an image setting `NVIDIA_DISABLE_REQUIRE`,
  skips the requirement check entirely. The second is detected and reported; the
  first is a property of the cluster, not the image.

Only a boot test closes these, and that tier is deferred (YP6M-3479).

## Inputs

| input | default | |
|---|---|---|
| `images` | — | whitespace/newline-separated. Digest-pinned refs preferred. |
| `targets` | — | whitespace/newline-separated target classes. |
| `strict-unknown` | `false` | `true` makes CANNOT-VERIFY fail the step. INCOMPATIBLE always fails. |
| `report-path` | `cuda-compat-report.json` | where the JSON report is written. |

## Outputs

`verdict`, `exit-code`, `report` (path), and `summary` — a markdown table with
the digests actually verdicted, ready to paste into a PR or an issue body.

## Tests

```bash
cd tests
python3 -m unittest test_cuda_compat test_dsl_parity   # hermetic
python3 -m unittest test_resolve_targets               # needs pyyaml
python3 -m unittest test_registry_contract             # network: fixtures vs registry
./mutation_check.py                                    # proves the suite bites
./mutation_check.py --list                             # the catalogue, and the count
```

`mutation_check.py` seeds a defect at a time and requires each to turn the suite
red. It covers both ways of being confidently wrong: the **evaluator** judging an
image against a target incorrectly (the AND/OR inversion among them), and the
**resolver** judging it against the wrong target — a fully correct verdict about
a machine that does not exist reads exactly like a correct one. A surviving
mutation is a reported hole, not a pass.

The count lives in `--list`, not here: a number in prose is one more thing that
goes stale while reading as reviewed.

## Constraints this file must keep

`cuda-compat.py` is stdlib-only, credential-free, and anonymous-auth only, so it
stays runnable by a customer with nothing but `python3`. Registry credentials for
the jfrog mirror belong in `action.yml` if they are ever needed — never in the
script (YP6M-3480, parked).
