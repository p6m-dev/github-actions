#!/usr/bin/env python3
"""Resolver tests: does a VersionConfig become the right target, or a refusal?

Built on synthetic platform-versions trees rather than the real repo, so these
assertions stay true when the estate changes — the live repo is covered by
running the resolver in CI, which is a different question (does it still parse
what we actually have?).

Needs PyYAML, like the resolver itself. Skipped rather than failed when absent,
so the dependency-free suite stays runnable anywhere.
"""
import importlib.util, os, tempfile, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
# RESOLVER_MUTANT_PATH lets mutation_check.py point these tests at a deliberately
# broken copy, the same way MUTANT_PATH does for cuda-compat.py.
RESOLVER = (os.environ.get("RESOLVER_MUTANT_PATH")
            or os.path.join(os.path.dirname(HERE), "resolve-targets.py"))

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
{karpenter_azure}{provisioners}"""


def write_config(root, name, *, azure=True, chart="1.12.2", enabled="true", skus=("T4",),
                 driver_version="580.126.20", overrides=None, time_slicing=False,
                 cuda_driver_major=None, extends=None, provisioners=True):
    """Write one synthetic VersionConfig.

    azure=True renders a karpenter-azure addon, which is how the resolver infers
    the cloud (the chart's own `cloud` value is injected per cluster and never
    appears in platform-versions). provisioners=False omits the
    karpenter-provisioners entry entirely, which is what makes a child inherit
    the parent's whole values blob rather than shadowing it with its own.
    """
    gpu_extra = ""
    if skus is not None:
        gpu_extra += "              skuGpuNames:\n"
        gpu_extra += "".join(f"                - {s}\n" for s in skus)
    if driver_version is not None and azure:
        gpu_extra += f'              driverVersion: "{driver_version}"\n'
    if cuda_driver_major is not None:
        gpu_extra += f'              cudaDriverMajor: "{cuda_driver_major}"\n'
    if overrides:
        gpu_extra += "              driverVersionOverrides:\n"
        gpu_extra += "".join(f'                {p}: "{v}"\n' for p, v in overrides.items())
    if time_slicing:
        gpu_extra += "              timeSlicing:\n                enabled: true\n"
    block = ""
    if provisioners:
        block = ("        karpenter-provisioners:\n"
                 "          values: |\n"
                 "            gpu:\n"
                 f"              enabled: {enabled}\n"
                 f"{gpu_extra}")
    body = CONFIG.format(
        name=name,
        extend=f"  extendRef:\n    name: {extends}\n" if extends else "",
        karpenter_azure=(f'        karpenter-azure:\n          targetRevision: "{chart}"\n'
                         if azure and chart else ""),
        provisioners=block)
    with open(os.path.join(root, f"{name}.yaml"), "w") as f:
        f.write(body)


INSTALLATION = """\
apiVersion: p6m.dev/v1alpha1
kind: Installation
metadata:
  name: nvidia-operator
spec:
  destinations:
    - clusterRef:
        name: some-cluster
      overrides:
        source:
          template: |
            nvidiaDrivers:
{entries}"""


def write_driver_crs(root, versions):
    entries = "".join(f'              - name: cr-{i}\n                version: "{v}"\n'
                      for i, v in enumerate(versions))
    with open(os.path.join(root, "nvidia-operator.yaml"), "w") as f:
        f.write(INSTALLATION.format(entries=entries))


@unittest.skipUnless(HAVE_YAML, "PyYAML not installed")
class TestResolver(unittest.TestCase):
    def setUp(self):
        self.mod = load_resolver()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.addCleanup(self.tmp.cleanup)

    def resolve(self, driver_crs=None):
        return self.mod.resolve(self.root, driver_crs)

    # --- the driver fact -------------------------------------------------

    def test_gpu_pool_becomes_a_target(self):
        write_config(self.root, "playground")
        resolved, unresolved = self.resolve()
        self.assertEqual(unresolved, [])
        self.assertEqual(resolved[0]["target"],
                         "playground=580.126.20/7.5/amd64/Tesla,Nvidia")

    def test_driver_comes_from_the_pool_declaration_not_the_karpenter_azure_pin(self):
        """Regression guard for YP6M-3479's original modelling error.

        Azure GPU nodes take AKSNodeClass spec.gpu.mode: None and the
        gpu-operator installs the DECLARED version. Deriving the driver from the
        karpenter-azure pin named the provider's compiled-in constant instead —
        a driver no node runs — and did it at exit 0. karpenter-azure 1.12.2
        compiles in 550.144.03; nothing here may produce it.
        """
        write_config(self.root, "pool", chart="1.12.2", driver_version="580.126.20")
        resolved, _ = self.resolve()
        self.assertEqual(resolved[0]["driver"], "580.126.20")
        self.assertNotIn("550", resolved[0]["target"])
        self.assertEqual(resolved[0]["driver_source"], "gpu.driverVersion")

    def test_each_rendered_pool_is_its_own_target_class(self):
        """An override is a whole pool at another driver, not a footnote."""
        write_config(self.root, "playground", driver_version="580.126.20",
                     time_slicing=True, overrides={"time-sliced-gpu": "570.211.01"})
        resolved, unresolved = self.resolve()
        self.assertEqual(unresolved, [])
        self.assertEqual({r["label"]: r["driver"] for r in resolved},
                         {"playground-gpu": "580.126.20",
                          "playground-time-sliced-gpu": "570.211.01"})

    def test_time_sliced_pool_does_not_exist_unless_enabled(self):
        write_config(self.root, "solo", time_slicing=False)
        resolved, _ = self.resolve()
        self.assertEqual([r["pool"] for r in resolved], ["gpu"])

    def test_override_on_a_pool_that_does_not_render_is_refused(self):
        """The chart fails the render here, so no such cluster exists to verdict."""
        write_config(self.root, "bad", time_slicing=False,
                     overrides={"time-sliced-gpu": "570.211.01"})
        resolved, unresolved = self.resolve()
        self.assertEqual(resolved, [])
        self.assertIn("does not render", unresolved[0]["reason"])

    def test_driver_major_alone_is_not_a_version(self):
        """"580" resolves to no nvcr.io/nvidia/driver image, so it installs nothing."""
        write_config(self.root, "partial", driver_version="580")
        resolved, unresolved = self.resolve()
        self.assertEqual(resolved, [])
        self.assertIn("gpu.driverVersion", unresolved[0]["reason"])

    def test_missing_driver_version_is_refused_not_defaulted(self):
        write_config(self.root, "silent", driver_version="")
        resolved, unresolved = self.resolve()
        self.assertEqual(resolved, [])
        self.assertIn("full NVIDIA driver version", unresolved[0]["reason"])

    def test_malformed_override_is_refused(self):
        write_config(self.root, "bad-override", time_slicing=True,
                     overrides={"time-sliced-gpu": "570"})
        resolved, unresolved = self.resolve()
        self.assertEqual(resolved, [])
        self.assertIn("driverVersionOverrides[time-sliced-gpu]", unresolved[0]["reason"])

    # --- AWS -------------------------------------------------------------

    def test_aws_pool_takes_the_declared_major_as_a_branch_only_target(self):
        """On AWS the AMI supplies the driver, so only the major is declared."""
        write_config(self.root, "aws-pool", azure=False, cuda_driver_major="550")
        resolved, unresolved = self.resolve()
        self.assertEqual(unresolved, [])
        self.assertEqual(resolved[0]["driver"], "550")
        self.assertEqual(resolved[0]["cloud"], "aws")
        self.assertEqual(resolved[0]["driver_source"], "gpu.cudaDriverMajor")

    def test_aws_pool_without_a_declared_major_is_refused(self):
        write_config(self.root, "aws-silent", azure=False)
        resolved, unresolved = self.resolve()
        self.assertEqual(resolved, [])
        self.assertIn("gpu.cudaDriverMajor", unresolved[0]["reason"])

    def test_driver_version_overrides_on_aws_are_refused(self):
        """Only Azure nodes take an operator-installed driver; the chart fails too."""
        write_config(self.root, "aws-override", azure=False, cuda_driver_major="550",
                     overrides={"gpu": "580.126.20"})
        resolved, unresolved = self.resolve()
        self.assertEqual(resolved, [])
        self.assertIn("non-Azure", unresolved[0]["reason"])

    # --- GPU models ------------------------------------------------------

    def test_absent_sku_list_is_refused_rather_than_defaulted(self):
        """The chart's default lives in another repo; a copy here is what rots.

        It moved A10 -> T4 in the same release that changed the driver model,
        which would have silently swapped every inheriting pool's compute
        capability from 8.6 to 7.5.
        """
        write_config(self.root, "inherits-default", skus=None)
        resolved, unresolved = self.resolve()
        self.assertEqual(resolved, [])
        self.assertIn("no gpu.skuGpuNames", unresolved[0]["reason"])

    def test_multi_sku_pool_becomes_several_target_classes(self):
        write_config(self.root, "mixed", skus=("T4", "V100"))
        resolved, _ = self.resolve()
        self.assertEqual([r["compute_capability"] for r in resolved], ["7.5", "7.0"])
        self.assertEqual([r["label"] for r in resolved], ["mixed-t4", "mixed-v100"])

    def test_grid_model_on_azure_is_refused_as_unrenderable(self):
        """The vanilla nvcr.io/nvidia/driver container cannot install on a
        converged/GRID model, and karpenter-provisioners fails the render."""
        write_config(self.root, "a10-pool", skus=("A10",))
        resolved, unresolved = self.resolve()
        self.assertEqual(resolved, [])
        self.assertIn("converged/GRID", unresolved[0]["reason"])

    def test_unknown_gpu_model_is_refused(self):
        write_config(self.root, "exotic", skus=("H200",))
        resolved, unresolved = self.resolve()
        self.assertEqual(resolved, [])
        self.assertIn("not in tables/gpu-compute-capability.yaml", unresolved[0]["reason"])

    def test_impossible_driver_and_silicon_pair_is_refused(self):
        """Blackwell on a 550 driver. Forward compat cannot carry a driver forward."""
        write_config(self.root, "impossible", skus=("B200",), driver_version="550.144.03")
        resolved, unresolved = self.resolve()
        self.assertEqual(resolved, [])
        self.assertIn("impossible pair", unresolved[0]["reason"])
        self.assertIn("needs driver 570+", unresolved[0]["reason"])

    def test_brand_set_is_a_superset_not_a_guess(self):
        """Unanimity across brands is only sound if the set covers every value
        the GPU may report; under-listing is the fail-open direction."""
        write_config(self.root, "t4", skus=("T4",))
        resolved, _ = self.resolve()
        self.assertIn("Tesla", resolved[0]["brands"])
        self.assertIn("Nvidia", resolved[0]["brands"])

    # --- the NVIDIADriver CR pair ----------------------------------------

    def test_declared_version_with_no_installing_cr_is_refused(self):
        """The failure this catches strands a pool: nodes come up labelled for a
        driver nothing delivers, and GPU pods pend forever. Declaring a version
        and installing it are two values in two repos with nothing enforcing it.
        """
        write_config(self.root, "stranded", driver_version="580.126.20")
        crs = tempfile.TemporaryDirectory()
        self.addCleanup(crs.cleanup)
        write_driver_crs(crs.name, ["570.211.01"])
        resolved, unresolved = self.resolve(driver_crs=crs.name)
        self.assertEqual(resolved, [])
        self.assertIn("no NVIDIADriver CR installs", unresolved[0]["reason"])

    def test_declared_version_with_a_matching_cr_resolves(self):
        write_config(self.root, "paired", driver_version="580.126.20")
        crs = tempfile.TemporaryDirectory()
        self.addCleanup(crs.cleanup)
        write_driver_crs(crs.name, ["580.126.20", "570.211.01"])
        resolved, unresolved = self.resolve(driver_crs=crs.name)
        self.assertEqual(unresolved, [])
        self.assertTrue(resolved[0]["cr_installed"])

    def test_cr_check_is_not_run_when_no_crs_are_supplied(self):
        """Absent input must not read as a passing check."""
        write_config(self.root, "unchecked", driver_version="580.126.20")
        resolved, _ = self.resolve()
        self.assertIsNone(resolved[0]["cr_installed"])

    # --- structure -------------------------------------------------------

    def test_non_gpu_pool_produces_nothing_at_all(self):
        write_config(self.root, "cpu-only", enabled="false")
        self.assertEqual(self.resolve(), ([], []))

    def test_values_are_inherited_through_extendref(self):
        write_config(self.root, "parent")
        write_config(self.root, "child", chart=None, provisioners=False, extends="parent")
        resolved, unresolved = self.resolve()
        self.assertEqual(unresolved, [])
        labels = {r["config"]: r["driver"] for r in resolved}
        self.assertEqual(labels["child"], "580.126.20",
                         "child inherits the parent's karpenter-provisioners values")

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
        prov = self.resolve()[0][0]["provenance"]
        self.assertIn("gpu.driverVersion", prov)
        self.assertIn("skuGpuNames T4", prov)


@unittest.skipUnless(HAVE_YAML, "PyYAML not installed")
class TestTables(unittest.TestCase):
    """The GPU table is the one hand-maintained fact set left; keep it honest."""

    def setUp(self):
        self.mod = load_resolver()

    def test_every_gpu_row_carries_the_facts_a_verdict_needs(self):
        gpus = self.mod.load_gpus()
        self.assertGreater(len(gpus), 5)
        for name, spec in gpus.items():
            with self.subTest(gpu=name):
                for field in ("compute_capability", "brands", "min_driver_branch"):
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
        for name, gpu in self.mod.load_gpus().items():
            for brand in gpu["brands"]:
                with self.subTest(gpu=name, brand=brand):
                    self.assertIn(brand.lower(), known,
                                  f"{brand} is not a string libnvidia-container can produce")

    def test_driver_family_is_only_ever_grid_or_absent(self):
        """The field's only remaining job is marking a model Azure cannot pool.

        It no longer selects a driver version — that comes from the pool's own
        declaration — so a stray value would silently stop refusing.
        """
        marked = 0
        for name, gpu in self.mod.load_gpus().items():
            family = gpu.get("driver_family")
            if family is None:
                continue
            marked += 1
            with self.subTest(gpu=name):
                self.assertEqual(family, "grid",
                                 f"{name}: 'cuda' is the absence of this field, not a value — "
                                 f"a row carrying it would never be refused")
        self.assertTrue(marked, "no model is marked grid; the refusal path is unreachable")


if __name__ == "__main__":
    unittest.main(verbosity=2)
