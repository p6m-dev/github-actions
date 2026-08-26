#!/usr/bin/env python3
"""Resolver tests: does a VersionConfig become the right target, or a refusal?

Built on synthetic platform-versions trees rather than the real repo, so these
assertions stay true when the estate changes — the live repo is covered by
running the resolver in CI, which is a different question (does it still parse
what we actually have?).

Needs PyYAML, like the resolver itself. Skipped rather than failed when absent,
so the dependency-free suite stays runnable anywhere.
"""
import importlib.util, os, tempfile, textwrap, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
RESOLVER = os.path.join(os.path.dirname(HERE), "resolve-targets.py")

try:
    import yaml  # noqa: F401
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False


def load_resolver():
    spec = importlib.util.spec_from_file_location("resolve_targets", RESOLVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CONFIG = """\
---
apiVersion: meta.p6m.dev/v1alpha1
kind: VersionConfig
metadata:
  name: {name}
spec:
{extend}  kubernetes:
    addons:
      helm:
{karpenter_azure}        karpenter-provisioners:
          values: |
            gpu:
              enabled: {enabled}
{gpu_extra}"""


def write_config(root, name, *, chart="1.12.2", enabled="true", skus=("T4",),
                 declared=None, extends=None):
    gpu_extra = ""
    if skus is not None:
        gpu_extra += "              skuGpuNames:\n"
        gpu_extra += "".join(f"                - {s}\n" for s in skus)
    if declared:
        gpu_extra += f'              cudaDriverMajor: "{declared}"\n'
    body = CONFIG.format(
        name=name,
        extend=f"  extendRef:\n    name: {extends}\n" if extends else "",
        karpenter_azure=(f'        karpenter-azure:\n          targetRevision: "{chart}"\n'
                         if chart else ""),
        enabled=enabled,
        gpu_extra=gpu_extra)
    with open(os.path.join(root, f"{name}.yaml"), "w") as f:
        f.write(body)


@unittest.skipUnless(HAVE_YAML, "PyYAML not installed")
class TestResolver(unittest.TestCase):
    def setUp(self):
        self.mod = load_resolver()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.addCleanup(self.tmp.cleanup)

    def resolve(self):
        return self.mod.resolve(self.root)

    def test_gpu_pool_becomes_a_target(self):
        write_config(self.root, "playground")
        resolved, unresolved = self.resolve()
        self.assertEqual(unresolved, [])
        self.assertEqual(resolved[0]["target"],
                         "playground=550.144.03/7.5/amd64/Tesla,Nvidia")

    def test_non_gpu_pool_produces_nothing_at_all(self):
        write_config(self.root, "cpu-only", enabled="false")
        self.assertEqual(self.resolve(), ([], []))

    def test_sku_list_defaults_to_the_chart_default(self):
        """A GPU pool that names no SKU still resolves — via the chart's A10 default."""
        write_config(self.root, "inherits-default", skus=None)
        resolved, _ = self.resolve()
        self.assertEqual(resolved[0]["compute_capability"], "8.6")

    def test_multi_sku_pool_becomes_several_target_classes(self):
        """One pool provisioning two GPU models is two target classes, not one."""
        write_config(self.root, "mixed", skus=("T4", "V100"))
        resolved, _ = self.resolve()
        self.assertEqual([r["compute_capability"] for r in resolved], ["7.5", "7.0"])
        self.assertEqual([r["label"] for r in resolved], ["mixed-t4", "mixed-v100"])

    def test_pool_mixing_driver_families_is_refused(self):
        """T4 takes the CUDA driver, A10 the converged/GRID one. No node runs both."""
        write_config(self.root, "impossible-mix", skus=("T4", "A10"))
        resolved, unresolved = self.resolve()
        self.assertEqual(resolved, [])
        self.assertIn("driver families", unresolved[0]["reason"])

    def test_a10_pool_gets_the_converged_driver_not_the_cuda_one(self):
        """Every Azure A10 size is in karpenter-azure's ConvergedGPUDriverSizes.

        Handing an A10 pool the CUDA driver version is a silent modelling error:
        it names a driver no such node ever runs.
        """
        write_config(self.root, "a10-pool", skus=("A10",))
        resolved, _ = self.resolve()
        self.assertEqual(resolved[0]["driver"], "550.144.06")
        self.assertEqual(resolved[0]["driver_family"], "grid")
        self.assertIn("grid driver", resolved[0]["provenance"])

    def test_t4_pool_still_gets_the_cuda_driver(self):
        write_config(self.root, "t4-pool", skus=("T4",))
        resolved, _ = self.resolve()
        self.assertEqual(resolved[0]["driver"], "550.144.03")
        self.assertEqual(resolved[0]["driver_family"], "cuda")

    def test_brand_set_is_a_superset_not_a_guess(self):
        """Unanimity across brands is only sound if the set covers every value
        the GPU may report; under-listing is the fail-open direction."""
        write_config(self.root, "a10", skus=("A10",))
        resolved, _ = self.resolve()
        self.assertIn("GRID", resolved[0]["brands"])
        self.assertIn("Nvidia", resolved[0]["brands"])

    def test_driver_comes_from_the_chart_pin_not_the_declared_label(self):
        write_config(self.root, "pool", declared="550")
        resolved, _ = self.resolve()
        self.assertEqual(resolved[0]["driver"], "550.144.03")

    def test_declared_label_contradicting_the_pin_is_refused(self):
        write_config(self.root, "drifted", declared="535")
        resolved, unresolved = self.resolve()
        self.assertEqual(resolved, [])
        self.assertIn("declared gpu.cudaDriverMajor=535", unresolved[0]["reason"])
        self.assertIn("550.144.03", unresolved[0]["reason"])

    def test_unknown_chart_version_is_refused(self):
        write_config(self.root, "future", chart="9.9.9")
        resolved, unresolved = self.resolve()
        self.assertEqual(resolved, [])
        self.assertIn("not in tables/karpenter-azure-driver.yaml", unresolved[0]["reason"])

    def test_unknown_gpu_model_is_refused(self):
        write_config(self.root, "exotic", skus=("H200",))
        resolved, unresolved = self.resolve()
        self.assertEqual(resolved, [])
        self.assertIn("not in tables/gpu-compute-capability.yaml", unresolved[0]["reason"])

    def test_impossible_driver_and_silicon_pair_is_refused(self):
        """Blackwell on a 550 driver. Forward compat cannot carry a driver forward."""
        write_config(self.root, "impossible", skus=("B200",))
        resolved, unresolved = self.resolve()
        self.assertEqual(resolved, [])
        self.assertIn("impossible pair", unresolved[0]["reason"])
        self.assertIn("needs driver 570+", unresolved[0]["reason"])

    def test_missing_chart_pin_is_refused_not_defaulted(self):
        write_config(self.root, "unpinned", chart=None)
        resolved, unresolved = self.resolve()
        self.assertEqual(resolved, [])
        self.assertIn("no karpenter-azure targetRevision", unresolved[0]["reason"])

    def test_values_are_inherited_through_extendref(self):
        write_config(self.root, "parent")
        write_config(self.root, "child", chart=None, enabled="true", skus=None,
                     extends="parent")
        resolved, unresolved = self.resolve()
        self.assertEqual(unresolved, [])
        labels = {r["label"]: r["driver"] for r in resolved}
        self.assertEqual(labels["child"], "550.144.06",
                         "child inherits the parent's karpenter-azure pin (A10 -> converged driver)")

    def test_extendref_cycle_does_not_hang(self):
        write_config(self.root, "a", chart=None, extends="b")
        write_config(self.root, "b", chart=None, extends="a")
        resolved, unresolved = self.resolve()
        self.assertEqual(resolved, [])
        self.assertEqual(len(unresolved), 2)

    def test_duplicate_config_names_are_refused(self):
        write_config(self.root, "twin")
        os.makedirs(os.path.join(self.root, "other"))
        write_config(os.path.join(self.root, "other"), "twin")
        with self.assertRaises(self.mod.DuplicateConfig):
            self.resolve()

    def test_provenance_names_both_sources(self):
        write_config(self.root, "pool")
        resolved, _ = self.resolve()
        prov = resolved[0]["provenance"]
        self.assertIn("karpenter-azure 1.12.2", prov)
        self.assertIn("skuGpuNames T4", prov)


@unittest.skipUnless(HAVE_YAML, "PyYAML not installed")
class TestTables(unittest.TestCase):
    """The tables are hand-maintained facts; keep their shape honest."""

    def setUp(self):
        self.mod = load_resolver()

    def test_every_gpu_row_carries_the_facts_a_verdict_needs(self):
        gpus = self.mod.load_table(self.mod.GPU_TABLE, "gpus")
        self.assertGreater(len(gpus), 5)
        for name, spec in gpus.items():
            with self.subTest(gpu=name):
                for field in ("compute_capability", "brands", "min_driver_branch", "reviewed"):
                    self.assertIn(field, spec, f"{name} is missing {field}")
                self.assertTrue(spec["brands"], f"{name} lists no NVML brand")
                self.assertRegex(spec["compute_capability"], r"^\d+\.\d+$")
                self.assertRegex(str(spec["min_driver_branch"]), r"^\d+$")

    def test_brands_are_real_nvml_strings(self):
        """A typo'd brand would quietly never match, turning verdicts undecidable."""
        checker_path = os.path.join(os.path.dirname(HERE), "cuda-compat.py")
        spec = importlib.util.spec_from_file_location("cuda_compat", checker_path)
        cc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cc)
        known = {b.lower() for b in cc.NVML_BRANDS}
        for name, gpu in self.mod.load_table(self.mod.GPU_TABLE, "gpus").items():
            for brand in gpu["brands"]:
                with self.subTest(gpu=name, brand=brand):
                    self.assertIn(brand.lower(), known,
                                  f"{brand} is not a string libnvidia-container can produce")

    def test_every_driver_row_records_where_it_was_read_from(self):
        for version, row in self.mod.load_table(self.mod.DRIVER_TABLE, "versions").items():
            with self.subTest(version=version):
                self.assertIn("cuda", row)
                self.assertRegex(row["cuda"], r"^\d+\.\d+")
                self.assertIn(f"v{version}", row["source"],
                              "source link must point at the tag the constant was read from")


if __name__ == "__main__":
    unittest.main(verbosity=2)
