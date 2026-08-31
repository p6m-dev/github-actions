#!/usr/bin/env python3
"""Derive cuda-compat target classes from platform-versions. No hand registration.

Every GPU-enabled VersionConfig becomes one target class per rendered POOL per
GPU model:

  Azure: gpu.driverVersion, narrowed per pool by gpu.driverVersionOverrides
  AWS:   gpu.cudaDriverMajor (branch only — the AMI decides the rest)
  both:  gpu.skuGpuNames --(GPU table)--> compute capability + NVML brands

so a new pool is covered the moment its VersionConfig merges, and nobody has to
remember to add it anywhere.

Why the driver comes from the pool's own declaration
----------------------------------------------------
Azure GPU nodes are provisioned with NO baked driver (AKSNodeClass
`spec.gpu.mode: None`), and the gpu-operator installs the version each pool
declares. So `gpu.driverVersion` — narrowed by `gpu.driverVersionOverrides` — is
what actually runs, and it is also what labels the pool
`p6m.dev/cuda.driver-version`.

Reading the driver out of the karpenter-azure pin instead was correct until
YP6M-3563/YP6M-3575 and is now the wrong fact: it names the provider's
compiled-in constant, which no node receives. That is worse than a refusal —
it verdicts confidently, at exit 0, against a driver the estate does not run,
and it cannot see a per-pool override at all. Deriving from the declaration
also deleted a hand-maintained table and its per-bump ritual.

Pool names and the override rules mirror `p6m.gpu.poolNames` and
`p6m.gpu.validateDriverOverrides` in the karpenter-provisioners chart. Where
that chart fails the render, this refuses the target: a config that cannot
render has no pool to verdict.

  ./resolve-targets.py --platform-versions ~/orgs/p6m-run/platform-versions
  ./resolve-targets.py --platform-versions ... --driver-crs ~/orgs/ybor-playground/.platform
  ./resolve-targets.py --platform-versions ... --json

Needs PyYAML (CI-side tool). cuda-compat.py itself stays stdlib-only.
"""
import argparse, glob, json, os, re, sys

try:
    import yaml
except ImportError:
    sys.exit("resolve-targets.py needs PyYAML (pip install pyyaml). "
             "cuda-compat.py itself has no dependencies; this resolver is CI-side.")

HERE = os.path.dirname(os.path.abspath(__file__))
GPU_TABLE = os.path.join(HERE, "tables", "gpu-compute-capability.yaml")

# The estate is x86 everywhere; an arm64 GPU pool would need its own row here
# rather than a default that quietly mislabels one.
POOL_ARCH = "amd64"

# Same shape the chart enforces: a major alone resolves to no nvcr.io/nvidia
# driver image, so it is not a version anything can install.
DRIVER_VERSION = re.compile(r"^\d{3,4}\.\d+\.\d+$")

# The pools templates/azure/nodepools.yaml renders, in render order. The only
# keys driverVersionOverrides may use.
BASE_POOL = "gpu"
TIME_SLICED_POOL = "time-sliced-gpu"


def load_gpus():
    with open(GPU_TABLE) as f:
        return yaml.safe_load(f)["gpus"]


class DuplicateConfig(Exception):
    """Two VersionConfigs share a name: which one runs is not knowable from here."""


def version_configs(repo):
    """Every VersionConfig in the repo, by metadata.name."""
    out = {}
    for path in sorted(glob.glob(os.path.join(repo, "**", "*.yaml"), recursive=True)):
        try:
            with open(path) as f:
                docs = list(yaml.safe_load_all(f))
        except yaml.YAMLError:
            continue
        for doc in docs:
            if isinstance(doc, dict) and doc.get("kind") == "VersionConfig":
                name = (doc.get("metadata") or {}).get("name")
                if not name:
                    continue
                rel = os.path.relpath(path, repo)
                if name in out and out[name]["path"] != rel:
                    # Last-one-wins would silently verdict one cluster's images
                    # against another cluster's driver.
                    raise DuplicateConfig(
                        f"VersionConfig '{name}' is defined in both {out[name]['path']} and "
                        f"{rel} — refusing to pick one")
                out[name] = {"doc": doc, "path": rel}
    return out


def addon(config, configs, name, field, _seen=None):
    """Read addons.helm.<name>.<field>, walking extendRef when the child is silent.

    Only whole fields are inherited. Merging two `values` blobs would mean
    guessing how a partial override composes, and a wrong guess here silently
    changes which driver a pool is judged against.
    """
    _seen = _seen or set()
    doc = config["doc"]
    if id(doc) in _seen:
        return None, None
    _seen.add(id(doc))
    helm = (((doc.get("spec") or {}).get("kubernetes") or {}).get("addons") or {}).get("helm") or {}
    entry = helm.get(name) or {}
    if field in entry:
        return entry[field], config["path"]
    parent = ((doc.get("spec") or {}).get("extendRef") or {}).get("name")
    if parent and parent in configs:
        return addon(configs[parent], configs, name, field, _seen)
    return None, None


def has_addon(config, configs, name, _seen=None):
    """Is this addon present at all (here or inherited)?"""
    _seen = _seen or set()
    doc = config["doc"]
    if id(doc) in _seen:
        return False
    _seen.add(id(doc))
    helm = (((doc.get("spec") or {}).get("kubernetes") or {}).get("addons") or {}).get("helm") or {}
    if name in helm:
        return True
    parent = ((doc.get("spec") or {}).get("extendRef") or {}).get("name")
    return bool(parent and parent in configs and has_addon(configs[parent], configs, name, _seen))


def gpu_values(config, configs):
    raw, source = addon(config, configs, "karpenter-provisioners", "values")
    if not raw:
        return None, None
    try:
        parsed = yaml.safe_load(raw) or {}
    except yaml.YAMLError as e:
        return {"_parse_error": str(e)}, source
    return parsed.get("gpu") or {}, source


def pool_names(gpu):
    """Mirrors p6m.gpu.poolNames: the base pool, plus the time-sliced one."""
    names = [BASE_POOL]
    if ((gpu.get("timeSlicing") or {}).get("enabled")):
        names.append(TIME_SLICED_POOL)
    return names


def azure_pool_drivers(gpu):
    """{pool: full driver version} for an Azure config, or a refusal string.

    Mirrors p6m.gpu.validateDriverOverrides / p6m.gpu.poolDriverVersion. Every
    way of declaring nothing is refused rather than ignored — silence is how a
    pool ends up wearing a version it does not run.
    """
    base = str(gpu.get("driverVersion") or "")
    if not DRIVER_VERSION.match(base):
        return None, (
            f"gpu.driverVersion is {base!r}, not a full NVIDIA driver version like "
            f"'580.126.20'. Azure GPU nodes carry no baked driver (AKSNodeClass "
            f"spec.gpu.mode: None); the gpu-operator installs exactly this string as the "
            f"nvcr.io/nvidia/driver tag, so there is no driver fact to verdict against. "
            f"The karpenter-provisioners chart fails the render on the same condition.")

    pools = pool_names(gpu)
    overrides = gpu.get("driverVersionOverrides") or {}
    if not isinstance(overrides, dict):
        return None, "gpu.driverVersionOverrides is not a mapping of pool name to version"

    for pool, version in overrides.items():
        if pool not in pools:
            return None, (
                f"gpu.driverVersionOverrides names pool {pool!r}, which does not render — "
                f"this config renders [{', '.join(pools)}]. {TIME_SLICED_POOL} renders only "
                f"with gpu.timeSlicing.enabled. The chart refuses this too, so there is no "
                f"cluster in this state to verdict.")
        if not DRIVER_VERSION.match(str(version)):
            return None, (f"gpu.driverVersionOverrides[{pool}] is {version!r}, not a full "
                          f"NVIDIA driver version")

    return {pool: str(overrides.get(pool, base)) for pool in pools}, None


def nvidia_driver_cr_versions(roots):
    """Every version an NVIDIADriver CR installs, across the Installations under roots.

    Set-based on purpose: an Installation names the clusters it targets, and a
    VersionConfig does not, so this cannot be matched per cluster from here. It
    still catches the failure that strands a pool — a declared version that no CR
    installs, whose nodes come up and never receive a driver.

    Takes several roots because the CRs live in each cluster org's own .platform
    repo, and GPU pools will not stay in one of them.
    """
    if isinstance(roots, str):
        roots = [roots]
    versions = {}
    paths = [(root, p) for root in roots
             for p in sorted(glob.glob(os.path.join(root, "**", "*.yaml"), recursive=True))]
    for root, path in paths:
        try:
            with open(path) as f:
                docs = list(yaml.safe_load_all(f))
        except (yaml.YAMLError, OSError):
            continue
        for doc in docs:
            if not isinstance(doc, dict) or doc.get("kind") != "Installation":
                continue
            for dest in ((doc.get("spec") or {}).get("destinations") or []):
                template = (((dest.get("overrides") or {}).get("source") or {})
                            .get("template") or "")
                if "nvidiaDrivers" not in template:
                    continue
                try:
                    values = yaml.safe_load(template) or {}
                except yaml.YAMLError:
                    continue
                for cr in (values.get("nvidiaDrivers") or []):
                    if isinstance(cr, dict) and cr.get("version"):
                        versions.setdefault(str(cr["version"]), set()).add(
                            os.path.relpath(path, root))
    return versions


def resolve(repo, driver_crs=None):
    gpus = load_gpus()
    configs = version_configs(repo)
    crs = nvidia_driver_cr_versions(driver_crs) if driver_crs else None

    resolved, unresolved = [], []
    for name in sorted(configs):
        config = configs[name]
        gpu, values_source = gpu_values(config, configs)
        if not gpu or not gpu.get("enabled"):
            continue  # not a GPU pool; nothing to verdict

        refuse = lambda why: unresolved.append(
            {"config": name, "path": config["path"], "reason": why})

        if "_parse_error" in gpu:
            refuse(f"karpenter-provisioners values is not parseable YAML: {gpu['_parse_error']}")
            continue

        # The chart's `cloud` value is injected per cluster and is not in
        # platform-versions, so infer it from the provisioner that is: only an
        # Azure cluster carries the karpenter-azure addon.
        azure = has_addon(config, configs, "karpenter-azure")

        # No chart default is reproduced here. Doing so duplicates a value that
        # lives in another repo, and the copy is what rots — the default moved
        # A10 -> T4 in the same release that changed the driver model, which
        # would have silently swapped every inheriting pool's compute capability.
        sku_names = gpu.get("skuGpuNames")
        if not sku_names:
            refuse("declares no gpu.skuGpuNames. The chart has a default, but reading it from "
                   "here would mean copying a value out of karpenter-provisioners and trusting "
                   "the copy — declare the models explicitly and the target is derived, not "
                   "guessed")
            continue
        unknown = [s for s in sku_names if s not in gpus]
        if unknown:
            refuse(f"GPU model(s) {', '.join(unknown)} not in tables/gpu-compute-capability.yaml — "
                   f"add one reviewed line per silicon generation")
            continue

        if azure:
            # A converged/GRID model cannot be an Azure GPU pool at all: the
            # vanilla nvcr.io/nvidia/driver container the NVIDIADriver CR uses
            # will not install on it, and the chart fails the render. Modelling
            # one would invent a pool that cannot exist.
            grid = [s for s in sku_names if gpus[s].get("driver_family") == "grid"]
            if grid:
                refuse(f"GPU model(s) {', '.join(grid)} take the converged/GRID driver, which the "
                       f"operator-installed vanilla driver cannot replace. karpenter-provisioners "
                       f"fails the render on this, so no such pool exists to verdict.")
                continue
            pool_drivers, why = azure_pool_drivers(gpu)
            if why:
                refuse(why)
                continue
            declared_by = "gpu.driverVersion"
        else:
            major = str(gpu.get("cudaDriverMajor") or "")
            if not re.fullmatch(r"\d{3,4}", major):
                refuse(f"gpu.cudaDriverMajor is {major!r}. On AWS the driver comes from the AMI, "
                       f"so the major is the only driver fact declared and there is nothing to "
                       f"derive it from. Read it off the AMI release notes, or off a node's "
                       f"nvidia.com/cuda.driver-version.major label.")
                continue
            if gpu.get("driverVersionOverrides"):
                refuse("gpu.driverVersionOverrides is set on a non-Azure config, where it does "
                       "nothing — only Azure GPU nodes take an operator-installed driver. The "
                       "chart fails the render on this.")
                continue
            # Branch-only. cuda-compat leaves any finer comparison undecided
            # rather than inventing a patch level.
            pool_drivers = {BASE_POOL: major}
            declared_by = "gpu.cudaDriverMajor"

        for pool, driver in pool_drivers.items():
            branch = driver.split(".")[0]

            if crs is not None and azure and driver not in crs:
                refuse(f"pool {pool!r} declares driver {driver}, which no NVIDIADriver CR "
                       f"installs (CRs found: {', '.join(sorted(crs)) or 'none'}). The pool's "
                       f"nodes would come up labelled for a driver nothing delivers and stay "
                       f"without one. Declaring a version and installing it are two values in "
                       f"two repos with nothing enforcing the pair.")
                continue

            for sku in sku_names:
                spec = gpus[sku]
                if int(branch) < int(spec["min_driver_branch"]):
                    refuse(f"impossible pair: {sku} (compute {spec['compute_capability']}, "
                           f"needs driver {spec['min_driver_branch']}+) on driver {driver} "
                           f"declared for pool {pool!r}. Forward compatibility carries a newer "
                           f"CUDA onto an older driver; it cannot carry a driver onto newer "
                           f"silicon.")
                    continue

                # A pool that can provision several GPU models is several target
                # classes, and two pools at two drivers are two more. Collapsing
                # either would verdict an image against a driver or a capability
                # no node has.
                parts = [name]
                if len(pool_drivers) > 1:
                    parts.append(pool)
                if len(sku_names) > 1:
                    parts.append(sku.lower())
                label = "-".join(parts)

                resolved.append({
                    "label": label,
                    "config": name,
                    "pool": pool,
                    "sku": sku,
                    "target": f"{label}={driver}/{spec['compute_capability']}/{POOL_ARCH}/"
                              + ",".join(spec["brands"]),
                    "driver": driver,
                    "driver_source": declared_by,
                    "cloud": "azure" if azure else "aws",
                    "compute_capability": spec["compute_capability"],
                    "brands": spec["brands"],
                    "cr_installed": None if crs is None else (driver in crs),
                    "provenance": (
                        f"{declared_by} -> pool {pool} runs driver {driver} ({values_source}); "
                        f"skuGpuNames {sku} -> compute {spec['compute_capability']}"),
                })
    return resolved, unresolved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform-versions", required=True,
                    help="path to a checkout of p6m-run/platform-versions")
    ap.add_argument("--driver-crs", metavar="PATH", action="append",
                    help="path to a checkout carrying the NVIDIADriver Installations "
                         "(e.g. ybor-playground/.platform). Repeatable. When given, a pool "
                         "declaring a version no CR installs is refused instead of resolved.")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--github-output", action="store_true",
                    help="append targets/unresolved to $GITHUB_OUTPUT")
    args = ap.parse_args()

    resolved, unresolved = resolve(args.platform_versions, args.driver_crs)

    if args.json:
        print(json.dumps({"resolved": resolved, "unresolved": unresolved}, indent=2))
    else:
        print(f"resolved {len(resolved)} target class(es) from GPU-enabled VersionConfigs:\n")
        for r in resolved:
            print(f"  {r['target']}")
            print(f"      {r['provenance']}")
        if not args.driver_crs:
            print("\n  (no --driver-crs given: not checked against the NVIDIADriver CRs that "
                  "install these versions)")
        if unresolved:
            print(f"\nCANNOT-VERIFY — {len(unresolved)} GPU pool(s) produced no target:\n")
            for u in unresolved:
                print(f"  {u['config']} ({u['path']})\n      {u['reason']}")

    if args.github_output and os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write("targets<<CC_EOF\n" + "\n".join(r["target"] for r in resolved) + "\nCC_EOF\n")
            f.write(f"resolved-count={len(resolved)}\n")
            f.write(f"unresolved-count={len(unresolved)}\n")
            # So a consumer can say "not checked" out loud rather than let the
            # guard's absence read as the guard having passed.
            f.write(f"cr-checked={'true' if args.driver_crs else 'false'}\n")
            f.write("unresolved<<CC_EOF\n"
                    + "\n".join(f"- **{u['config']}** (`{u['path']}`): {u['reason']}"
                                for u in unresolved) + "\nCC_EOF\n")

    # An estate with no resolvable target is not a pass — it is a checker with
    # nothing to check, and the caller must be able to tell those apart.
    if not resolved:
        print("\nno GPU target classes resolved — nothing would be verified", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
