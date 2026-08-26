#!/usr/bin/env python3
"""Record a digest-pinned fixture from a live image, for the offline suite.

  ./record_fixture.py NAME REF [--note "why this image is in the suite"]

Writes fixtures/NAME.json holding the image's index digest, every linux child
manifest, and only the CUDA-relevant env vars. Two reasons the suite records
rather than fetching at test time:

  - a registry tag move must never flake the suite (the fixture pins the digest
    it was recorded from, and test_registry_contract.py re-verifies that pin
    against the live registry — so drift is *reported*, never absorbed);
  - the evaluator tests stay hermetic, which is what makes the mutation check
    meaningful: a red run means the logic broke, not that the network did.

Re-record deliberately, never to make a red suite green.
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib.util

spec = importlib.util.spec_from_file_location(
    "cuda_compat", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cuda-compat.py"))
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)

# Only what the evaluator reads. Recording the whole env would put unrelated
# (and potentially sensitive) image state in git for no benefit. Every
# NVIDIA_REQUIRE_* var counts: the runtime ANDs them.
KEYS = ("CUDA_VERSION", "TORCH_CUDA_ARCH_LIST", "NVIDIA_DRIVER_CAPABILITIES",
        "NVIDIA_DISABLE_REQUIRE")


def keep(k):
    return k in KEYS or k.startswith("NVIDIA_REQUIRE_")

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def record(name, ref, note):
    index_digest, platforms = cc.fetch_platforms(ref)
    repo = ref.split("@")[0].rsplit(":", 1)[0] if "@" not in ref else ref.split("@")[0]
    fixture = {
        "recorded_from": ref,
        "pinned": f"{repo}@{index_digest}" if index_digest else ref,
        "index_digest": index_digest,
        "note": note,
        "platforms": [{"platform": p["platform"], "arch": p["arch"], "digest": p["digest"],
                       "env": {k: v for k, v in sorted(p["env"].items()) if keep(k)}}
                      for p in platforms],
    }
    path = os.path.join(FIXTURES, f"{name}.json")
    os.makedirs(FIXTURES, exist_ok=True)
    with open(path, "w") as f:
        json.dump(fixture, f, indent=2, sort_keys=False)
        f.write("\n")
    print(f"wrote {path}\n  pinned: {fixture['pinned']}")
    for p in fixture["platforms"]:
        print(f"  {p['platform']}: {p['env'] or '(no CUDA metadata)'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("ref")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    record(a.name, a.ref, a.note)
