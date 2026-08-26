"""Shared test plumbing: import cuda-compat.py (a hyphenated script, not a module).

MUTANT_PATH lets mutation_check.py point the whole suite at a deliberately
broken copy of the script without touching the tests themselves.
"""
import importlib.util, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.environ.get("MUTANT_PATH") or os.path.join(os.path.dirname(HERE), "cuda-compat.py")
FIXTURES = os.path.join(HERE, "fixtures")


def load_module():
    spec = importlib.util.spec_from_file_location("cuda_compat", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fixture(name):
    with open(os.path.join(FIXTURES, f"{name}.json")) as f:
        return json.load(f)


def platform(fx, arch):
    for p in fx["platforms"]:
        if p["arch"] == arch:
            return p
    raise AssertionError(f"fixture has no {arch} platform: {[p['arch'] for p in fx['platforms']]}")
