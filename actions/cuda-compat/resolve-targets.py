#!/usr/bin/env python3
"""Derive cuda-compat target classes from platform-versions. No hand registration.

Every GPU-enabled VersionConfig becomes a target class:

  karpenter-azure.targetRevision --(driver table)--> the driver the pool ACTUALLY runs
  gpu.skuGpuNames               --(GPU table)-----> compute capability + NVML brands

so a new pool is covered the moment its VersionConfig merges, and nobody has to
remember to add it anywhere. Both tables are fail-honest: an entry that is not
there produces a refusal with a reason, never a guess and never a skip.

The driver comes from the karpenter-azure pin rather than the declared
gpu.cudaDriverMajor label, because the pin is what the provider compiles in. The
declaration is CROSS-CHECKED against it and a disagreement is reported — that
pairing is the whole point: one fact is what runs, the other is what we told
Kubernetes is running, and drift between them is a real incident class.

  ./resolve-targets.py --platform-versions ~/orgs/p6m-run/platform-versions
  ./resolve-targets.py --platform-versions ... --json
  ./resolve-targets.py --platform-versions ... --github-output

Needs PyYAML (CI-side tool). cuda-compat.py itself stays stdlib-only.
"""
import argparse, glob, json, os, sys

try:
    import yaml
except ImportError:
    sys.exit("resolve-targets.py needs PyYAML (pip install pyyaml). "
             "cuda-compat.py itself has no dependencies; this resolver is CI-side.")

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER_TABLE = os.path.join(HERE, "tables", "karpenter-azure-driver.yaml")
GPU_TABLE = os.path.join(HERE, "tables", "gpu-compute-capability.yaml")

# The estate is x86 everywhere; an arm64 GPU pool would need its own row here
# rather than a default that quietly mislabels one.
POOL_ARCH = "amd64"


def load_table(path, key):
    with open(path) as f:
        return yaml.safe_load(f)[key]


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


def gpu_values(config, configs):
    raw, source = addon(config, configs, "karpenter-provisioners", "values")
    if not raw:
        return None, None
    try:
        parsed = yaml.safe_load(raw) or {}
    except yaml.YAMLError as e:
        return {"_parse_error": str(e)}, source
    return parsed.get("gpu") or {}, source


def resolve(repo):
    drivers = load_table(DRIVER_TABLE, "versions")
    gpus = load_table(GPU_TABLE, "gpus")
    configs = version_configs(repo)

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

        chart, chart_source = addon(config, configs, "karpenter-azure", "targetRevision")
        if not chart:
            refuse("no karpenter-azure targetRevision (nor inherited) — the driver a GPU pool "
                   "runs is decided by that pin, so there is nothing to derive from")
            continue
        row = drivers.get(str(chart))
        if not row:
            refuse(f"karpenter-azure {chart} is not in tables/karpenter-azure-driver.yaml — "
                   f"add the row from that tag's pkg/utils/gpu.go; never interpolate")
            continue

        sku_names = gpu.get("skuGpuNames") or ["A10"]  # chart default
        unknown = [s for s in sku_names if s not in gpus]
        if unknown:
            refuse(f"GPU model(s) {', '.join(unknown)} not in tables/gpu-compute-capability.yaml — "
                   f"add one reviewed line per silicon generation")
            continue

        # Which driver the pool actually gets is a property of the SKU, not of the
        # chart: karpenter-azure installs the converged/GRID driver for the sizes
        # in ConvergedGPUDriverSizes and the CUDA driver otherwise. Every A10 size
        # is converged, so a pool of them runs a different driver VERSION than the
        # cuda one recorded against the same chart tag.
        families = {gpus[s].get("driver_family", "cuda") for s in sku_names}
        if len(families) > 1:
            refuse(f"pool mixes GPU models needing different driver families "
                   f"({', '.join(sorted(families))}); one pool cannot run both, so the "
                   f"declared skuGpuNames cannot describe a single node")
            continue
        family = families.pop()
        driver = row.get(family)
        if not driver:
            refuse(f"karpenter-azure {chart} records no '{family}' driver in "
                   f"tables/karpenter-azure-driver.yaml")
            continue

        # A pool that can provision several GPU models is several target classes:
        # collapsing them would verdict one image against a capability no node has.
        declared = str(gpu.get("cudaDriverMajor") or "") or None
        for sku in sku_names:
            spec = gpus[sku]
            branch = driver.split(".")[0]
            if int(branch) < int(spec["min_driver_branch"]):
                refuse(f"impossible pair: {sku} (compute {spec['compute_capability']}, "
                       f"needs driver {spec['min_driver_branch']}+) on driver {driver} from "
                       f"karpenter-azure {chart}. Forward compatibility carries a newer CUDA onto "
                       f"an older driver; it cannot carry a driver onto newer silicon.")
                continue

            contradiction = None
            if declared and declared != branch:
                contradiction = (f"declared gpu.cudaDriverMajor={declared} but karpenter-azure "
                                 f"{chart} installs {driver}. The label is what workloads schedule "
                                 f"against; the pin is what runs. Fix the declaration.")
                refuse(contradiction)
                continue

            label = name if len(sku_names) == 1 else f"{name}-{sku.lower()}"
            resolved.append({
                "label": label,
                "config": name,
                "sku": sku,
                "target": f"{label}={driver}/{spec['compute_capability']}/{POOL_ARCH}/"
                          + ",".join(spec["brands"]),
                "driver": driver,
                "driver_family": family,
                "compute_capability": spec["compute_capability"],
                "brands": spec["brands"],
                "declared_major": declared,
                "provenance": (f"karpenter-azure {chart} ({chart_source}) -> {family} driver {driver}; "
                               f"skuGpuNames {sku} ({values_source}) -> compute "
                               f"{spec['compute_capability']}"),
            })
    return resolved, unresolved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform-versions", required=True,
                    help="path to a checkout of p6m-run/platform-versions")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--github-output", action="store_true",
                    help="append targets/unresolved to $GITHUB_OUTPUT")
    args = ap.parse_args()

    resolved, unresolved = resolve(args.platform_versions)

    if args.json:
        print(json.dumps({"resolved": resolved, "unresolved": unresolved}, indent=2))
    else:
        print(f"resolved {len(resolved)} target class(es) from GPU-enabled VersionConfigs:\n")
        for r in resolved:
            print(f"  {r['target']}")
            print(f"      {r['provenance']}")
        if unresolved:
            print(f"\nCANNOT-VERIFY — {len(unresolved)} GPU pool(s) produced no target:\n")
            for u in unresolved:
                print(f"  {u['config']} ({u['path']})\n      {u['reason']}")

    if args.github_output and os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write("targets<<CC_EOF\n" + "\n".join(r["target"] for r in resolved) + "\nCC_EOF\n")
            f.write(f"resolved-count={len(resolved)}\n")
            f.write(f"unresolved-count={len(unresolved)}\n")
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
