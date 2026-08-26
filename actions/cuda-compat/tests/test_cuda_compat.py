#!/usr/bin/env python3
"""Offline suite for cuda-compat: verdicts, refusals, and the facts they rest on.

Hermetic — every image is a digest-pinned recording in fixtures/ (see
record_fixture.py). test_registry_contract.py is the online half that re-checks
those recordings against the registry; this file never touches the network, so a
red run here always means the logic changed.

The bar these tests are written to (inherited from YP6M-3482):
  - digest-pinned known-value images, so a tag move cannot flake the suite
  - BOTH children of an index image asserted — the single-request regression
    fails open on exactly the newest images
  - a seeded incompatible case that must be refused
  - unknown facts fail loudly, with negative controls asserting the loud failure
  - mutation-verified: see mutation_check.py
"""
import sys, unittest

from helpers import fixture, load_module, platform

cc = load_module()

# The estate's target classes, stated once. Driver versions are the full strings
# a node reports, because a bare branch cannot settle a finer comparison.
T4_POOL = "playground-t4=550.144.03/7.5/amd64/Tesla,Nvidia"
FLEX_NODE = "flex-blackwell=580.65.06/12.0/amd64/Nvidia"


def verdict(fixture_name, target_spec, arch="amd64"):
    fx = fixture(fixture_name)
    target = cc.parse_target(target_spec)
    worst, findings, basis = cc.verdict_for_platform(platform(fx, arch)["env"], target)
    return cc.LABELS[worst], basis, findings


def whole_image(fixture_name, target_spec):
    fx = fixture(fixture_name)
    return cc.evaluate(fx["pinned"], fx["platforms"], fx["index_digest"],
                       cc.parse_target(target_spec))


class TestStockRuntimeImages(unittest.TestCase):
    """llmkube's stock default runtime images against the estate's real pools.

    Each of these declares a CUDA newer than driver 550 provides, and each still
    runs there, because its requirement string ORs in the driver branches its
    bundled cuda-compat package supports. Read the separators the other way round
    and every one of these assertions flips — which is the point of pinning them
    to real images rather than to hand-written expressions.
    """

    def test_vllm_runs_on_the_t4_pools_via_forward_compat(self):
        label, basis, _ = verdict("vllm-openai-v0.25.1", T4_POOL)
        self.assertEqual(label, "COMPATIBLE")
        self.assertIn("forward compatibility", basis)
        self.assertIn("driver>=550", basis)

    def test_sglang_runs_on_the_t4_pools_via_forward_compat(self):
        label, basis, _ = verdict("sglang-v0.5.15", T4_POOL)
        self.assertEqual(label, "COMPATIBLE")
        self.assertIn("forward compatibility", basis)

    def test_llamacpp_runs_on_the_t4_pools_via_forward_compat(self):
        label, basis, _ = verdict("llamacpp-server-cuda-b10068", T4_POOL)
        self.assertEqual(label, "COMPATIBLE")
        self.assertIn("forward compatibility", basis)

    def test_all_three_are_natively_compatible_on_the_flex_node(self):
        for name in ("vllm-openai-v0.25.1", "sglang-v0.5.15", "llamacpp-server-cuda-b10068"):
            with self.subTest(image=name):
                label, basis, _ = verdict(name, FLEX_NODE)
                self.assertEqual(label, "COMPATIBLE")
                self.assertEqual(basis, "native CUDA-version match")

    def test_basis_distinguishes_forward_compat_from_native(self):
        """Both are COMPATIBLE; conflating them would hide a real operational risk."""
        _, t4_basis, _ = verdict("vllm-openai-v0.25.1", T4_POOL)
        _, flex_basis, _ = verdict("vllm-openai-v0.25.1", FLEX_NODE)
        self.assertNotEqual(t4_basis, flex_basis)
        self.assertIn("cuda-compat", t4_basis)


class TestIncompatibility(unittest.TestCase):
    def test_unenumerated_driver_branch_is_refused(self):
        """545 sits between two allowances: no term can carry it."""
        label, _, findings = verdict("vllm-openai-v0.25.1", "mid-branch=545.23.08/7.5/amd64/Tesla")
        self.assertEqual(label, "INCOMPATIBLE")
        detail = " ".join(f[2] for f in findings)
        self.assertIn("container refused at start", detail)
        self.assertIn("unsatisfied condition: cuda>=13.0", detail,
                      "the verdict must quote the message an operator sees in the log")

    def test_consumer_gpu_is_refused_on_the_same_driver(self):
        """Forward compat is datacenter-only. Same driver, different brand, different answer."""
        ok, _, _ = verdict("vllm-openai-v0.25.1", "t4=550.144.03/7.5/amd64/Tesla")
        bad, _, _ = verdict("vllm-openai-v0.25.1", "consumer=550.144.03/7.5/amd64/GeForce")
        self.assertEqual((ok, bad), ("COMPATIBLE", "INCOMPATIBLE"))

    def test_seeded_incompatible_image_is_refused(self):
        """Seeded defect: an image demanding a CUDA no driver in the table provides."""
        env = {"NVIDIA_REQUIRE_CUDA": "cuda>=99.0"}
        worst, findings, basis = cc.verdict_for_platform(env, cc.parse_target(T4_POOL))
        self.assertEqual(cc.LABELS[worst], "INCOMPATIBLE")
        self.assertIsNone(basis)
        self.assertIn("unsatisfied condition: cuda>=99.0", " ".join(f[2] for f in findings))

    def test_arch_mismatch_is_caught_separately_from_the_driver_check(self):
        """A 7.5-less arch list fails after start, not at it — the finding must say so."""
        env = {"NVIDIA_REQUIRE_CUDA": "cuda>=12.0", "TORCH_CUDA_ARCH_LIST": "8.0 9.0"}
        worst, findings, _ = cc.verdict_for_platform(env, cc.parse_target(T4_POOL))
        self.assertEqual(cc.LABELS[worst], "INCOMPATIBLE")
        arch_finding = [f for f in findings if "TORCH_CUDA_ARCH_LIST" in f[0]][0]
        self.assertFalse(arch_finding[1])
        self.assertIn("not enforced at start", arch_finding[0])

    def test_arch_accelerated_suffix_counts_as_compiled(self):
        """sm_120a is 12.0 plus extra instructions — not a different, unmatched arch."""
        env = {"NVIDIA_REQUIRE_CUDA": "cuda>=12.8", "TORCH_CUDA_ARCH_LIST": "9.0 10.0a 12.0a"}
        worst, findings, _ = cc.verdict_for_platform(env, cc.parse_target(FLEX_NODE))
        self.assertEqual(cc.LABELS[worst], "COMPATIBLE")
        detail = [f[2] for f in findings if "TORCH_CUDA_ARCH_LIST" in f[0]][0]
        self.assertIn("arch-accelerated", detail)

    def test_family_specific_suffix_counts_as_compiled(self):
        ok, detail = cc.check_arch("9.0 10.0f", "10.0")
        self.assertTrue(ok)
        self.assertIn("family-specific", detail)

    def test_suffix_does_not_make_a_different_arch_match(self):
        self.assertFalse(cc.check_arch("12.0a", "10.0")[0])

    def test_ptx_only_helps_forwards(self):
        """PTX from a NEWER arch cannot JIT onto an older GPU."""
        self.assertFalse(cc.check_arch("12.0+PTX", "7.5")[0])
        self.assertFalse(cc.check_arch("8.9+PTX", "8.0")[0])
        self.assertTrue(cc.check_arch("7.0+PTX", "7.5")[0])

    def test_ptx_comparison_uses_full_compute_capability(self):
        """8.9 vs 8.0 must not collapse to a major-version comparison."""
        self.assertFalse(cc.check_arch("8.9+PTX", "8.6")[0])
        self.assertTrue(cc.check_arch("8.6+PTX", "8.9")[0])

    def test_ptx_forward_compat_is_recognised_but_qualified(self):
        env = {"NVIDIA_REQUIRE_CUDA": "cuda>=12.0", "TORCH_CUDA_ARCH_LIST": "7.0 9.0+PTX"}
        worst, findings, _ = cc.verdict_for_platform(env, cc.parse_target(FLEX_NODE))
        self.assertEqual(cc.LABELS[worst], "COMPATIBLE")
        detail = [f[2] for f in findings if "TORCH_CUDA_ARCH_LIST" in f[0]][0]
        self.assertIn("JIT-forward-compatible", detail)
        self.assertIn("not guaranteed", detail)


class TestRefusals(unittest.TestCase):
    """Absence of evidence never becomes evidence."""

    def test_image_without_cuda_metadata_cannot_be_verified(self):
        label, basis, findings = verdict("alpine-3.20", T4_POOL)
        self.assertEqual(label, "CANNOT-VERIFY")
        self.assertIsNone(basis)
        self.assertIn("declares neither", " ".join(f[2] for f in findings))

    def test_unknown_driver_branch_is_refused_loudly(self):
        result = whole_image("vllm-openai-v0.25.1", "future=999.1.2/7.5/amd64/Tesla")
        self.assertEqual(result["verdict"], "CANNOT-VERIFY")
        self.assertEqual(result["exit"], 2)
        self.assertIn("never guess", result["refusal"])

    def test_unknown_brand_leaves_the_verdict_undecided(self):
        """The satisfying term needs a brand. Without one, refuse — do not assume."""
        label, _, findings = verdict("vllm-openai-v0.25.1", "nobrand=550.144.03/7.5")
        self.assertEqual(label, "CANNOT-VERIFY")
        self.assertIn("brand", " ".join(f[2] for f in findings).lower())

    def test_absent_compute_capability_cannot_satisfy_an_arch_term(self):
        """A target with no cc must not have arch= terms assumed-true for it."""
        env = {"NVIDIA_REQUIRE_CUDA": "arch>=8.0"}
        no_cc = cc.parse_target("t=550.144.03//amd64/Tesla")
        self.assertIsNone(no_cc.cc)
        worst, findings, _ = cc.verdict_for_platform(env, no_cc)
        self.assertEqual(cc.LABELS[worst], "CANNOT-VERIFY")
        self.assertIn("undecided", " ".join(f[2] for f in findings).lower())

    def test_branch_only_driver_cannot_settle_a_finer_comparison(self):
        facts = {"cuda": "12.4", "driver": "550", "arch": "7.5", "brand": "Tesla"}
        res = cc.eval_expression("driver>=550.54.14", facts)
        self.assertEqual(res["status"], cc.UNDECIDED)
        self.assertIn("full version", res["detail"])

    def test_image_with_no_variant_for_the_target_arch_is_refused(self):
        fx = fixture("vllm-openai-v0.25.1")
        amd64_only = [p for p in fx["platforms"] if p["arch"] == "amd64"]
        result = cc.evaluate(fx["pinned"], amd64_only, fx["index_digest"],
                             cc.parse_target("arm-target=550.144.03/7.5/arm64/Tesla"))
        self.assertEqual(result["verdict"], "CANNOT-VERIFY")
        self.assertIn("no linux/arm64 variant", result["refusal"])

    def test_disable_require_is_reported_not_silently_passed(self):
        env = {"NVIDIA_REQUIRE_CUDA": "cuda>=99.0", "NVIDIA_DISABLE_REQUIRE": "true"}
        worst, findings, _ = cc.verdict_for_platform(env, cc.parse_target(T4_POOL))
        self.assertEqual(cc.LABELS[worst], "CANNOT-VERIFY")
        self.assertIn("NVIDIA_DISABLE_REQUIRE", " ".join(f[2] for f in findings))

    def test_unknown_property_is_not_treated_as_satisfied(self):
        env = {"NVIDIA_REQUIRE_CUDA": "nosuchkey>=1"}
        worst, findings, _ = cc.verdict_for_platform(env, cc.parse_target(T4_POOL))
        self.assertNotEqual(cc.LABELS[worst], "COMPATIBLE")


class TestLegacyImages(unittest.TestCase):
    def test_cuda_version_synthesis_gates_on_require_cuda_alone(self):
        """The runtime's IsLegacy() reads ONLY NVIDIA_REQUIRE_CUDA, and APPENDS.

        Gating on "no NVIDIA_REQUIRE_* at all", or replacing instead of
        appending, drops a conjunct the runtime enforces — and a dropped
        conjunct can only ever make this tool more permissive than the runtime.
        """
        legacy = {"CUDA_VERSION": "12.9.1"}
        reqs, _ = cc.collect_requirements(legacy)
        self.assertEqual([e for _, e in reqs], ["cuda>=12.9"])

        modern = {"CUDA_VERSION": "12.9.1", "NVIDIA_REQUIRE_CUDA": "cuda>=12.0"}
        reqs, _ = cc.collect_requirements(modern)
        self.assertEqual([e for _, e in reqs], ["cuda>=12.0"])

        # A non-CUDA require var must NOT suppress the synthesis: the runtime
        # would evaluate both, ANDed.
        mixed = {"CUDA_VERSION": "12.6.2", "NVIDIA_REQUIRE_DRIVER": "driver>=470,driver<471"}
        reqs, _ = cc.collect_requirements(mixed)
        self.assertEqual(sorted(e for _, e in reqs), ["cuda>=12.6", "driver>=470,driver<471"])

    def test_dropped_conjunct_would_be_a_false_compatible(self):
        """The same image on a 470 node: the synthesized cuda>=12.6 must refuse it."""
        env = {"CUDA_VERSION": "12.6.2", "NVIDIA_REQUIRE_DRIVER": "driver>=470,driver<471"}
        worst, _, _ = cc.verdict_for_platform(
            env, cc.parse_target("old=470.82.01/7.5/amd64/Tesla"))
        self.assertEqual(cc.LABELS[worst], "INCOMPATIBLE")

    def test_legacy_image_too_new_for_the_driver_is_refused(self):
        worst, _, _ = cc.verdict_for_platform({"CUDA_VERSION": "12.9.1"}, cc.parse_target(T4_POOL))
        self.assertEqual(cc.LABELS[worst], "INCOMPATIBLE")

    def test_every_require_var_is_evaluated_not_just_cuda(self):
        env = {"NVIDIA_REQUIRE_CUDA": "cuda>=12.0", "NVIDIA_REQUIRE_DRIVER": "driver>=999"}
        worst, _, _ = cc.verdict_for_platform(env, cc.parse_target(T4_POOL))
        self.assertEqual(cc.LABELS[worst], "INCOMPATIBLE",
                         "NVIDIA_REQUIRE_* vars are ANDed; ignoring one fails open")

    def test_jetpack_requirement_is_excluded(self):
        env = {"NVIDIA_REQUIRE_JETPACK": "csv-mounts=all", "NVIDIA_REQUIRE_CUDA": "cuda>=12.0"}
        reqs, _ = cc.collect_requirements(env)
        self.assertEqual([e for _, e in reqs], ["cuda>=12.0"])


class TestIndexImages(unittest.TestCase):
    """Both children, always. Reading only the first fails open on new images."""

    def test_both_index_children_are_evaluated(self):
        result = whole_image("vllm-openai-v0.25.1", T4_POOL)
        platforms = {p["platform"]: p for p in result["platforms"]}
        self.assertEqual(set(platforms), {"linux/amd64", "linux/arm64"})
        for p in platforms.values():
            self.assertTrue(p["findings"], "a child manifest was listed but never evaluated")

    def test_children_have_distinct_digests(self):
        fx = fixture("vllm-openai-v0.25.1")
        digests = [p["digest"] for p in fx["platforms"]]
        self.assertEqual(len(digests), len(set(digests)))
        self.assertTrue(all(d.startswith("sha256:") for d in digests))

    def test_untargeted_child_cannot_change_the_verdict(self):
        """The arm64 child lacks 7.5 kernels; it must not vote on an amd64 node."""
        result = whole_image("vllm-openai-v0.25.1", T4_POOL)
        by_platform = {p["platform"]: p for p in result["platforms"]}
        self.assertEqual(by_platform["linux/arm64"]["verdict"], "INCOMPATIBLE")
        self.assertFalse(by_platform["linux/arm64"]["targeted"])
        self.assertEqual(result["verdict"], "COMPATIBLE")

    def test_fetch_platforms_reads_every_child_not_just_the_first(self):
        """The traversal itself, with a stub registry — no network, no recording.

        Recording a fixture would hide this: record_fixture.py calls the same
        function, so a first-child-only regression would quietly re-record as
        'this image has one platform' and every other test would still pass.
        """
        index = {"mediaType": cc.MEDIA_INDEX[0], "manifests": [
            {"digest": "sha256:amd", "platform": {"os": "linux", "architecture": "amd64"}},
            {"digest": "sha256:arm", "platform": {"os": "linux", "architecture": "arm64"}},
            {"digest": "sha256:att", "platform": {"os": "unknown", "architecture": "unknown"},
             "annotations": {"vnd.docker.reference.type": "attestation-manifest"}},
            {"digest": "sha256:win", "platform": {"os": "windows", "architecture": "amd64"}},
        ]}
        configs = {"sha256:amd": ["NVIDIA_REQUIRE_CUDA=cuda>=12.0"],
                   "sha256:arm": ["NVIDIA_REQUIRE_CUDA=cuda>=13.0"]}

        class StubRegistry:
            def __init__(self, registry, repo):
                pass

            def manifest(self, ref):
                if ref in configs:
                    return {"config": {"digest": f"cfg:{ref}"}}, None
                return index, "sha256:indexdigest"

            def blob(self, digest):
                return {"config": {"Env": configs[digest.split("cfg:")[1]]}}

        real, cc.Registry = cc.Registry, StubRegistry
        try:
            digest, platforms = cc.fetch_platforms("example.com/img:tag")
        finally:
            cc.Registry = real

        self.assertEqual(digest, "sha256:indexdigest")
        self.assertEqual([p["platform"] for p in platforms], ["linux/amd64", "linux/arm64"],
                         "must read both linux children, and neither the attestation nor windows")
        self.assertEqual(platforms[0]["env"]["NVIDIA_REQUIRE_CUDA"], "cuda>=12.0")
        self.assertEqual(platforms[1]["env"]["NVIDIA_REQUIRE_CUDA"], "cuda>=13.0")

    def test_fixtures_are_digest_pinned(self):
        for name in ("vllm-openai-v0.25.1", "sglang-v0.5.15", "llamacpp-server-cuda-b10068",
                     "cuda-12.4.1-base", "cuda-12.6.2-base", "alpine-3.20"):
            with self.subTest(fixture=name):
                fx = fixture(name)
                self.assertIn("@sha256:", fx["pinned"],
                              "a tag-pinned fixture would let a registry move flake the suite")


class TestTargets(unittest.TestCase):
    def test_target_parsing_fields(self):
        t = cc.parse_target("pool=550.144.03/7.5/arm64/Tesla,Nvidia")
        self.assertEqual((t.label, t.driver, t.branch, t.cc, t.arch),
                         ("pool", "550.144.03", "550", "7.5", "arm64"))
        self.assertEqual(t.brands, ["Tesla", "Nvidia"])

    def test_target_defaults_to_amd64(self):
        self.assertEqual(cc.parse_target("550").arch, "amd64")

    def test_empty_fields_are_allowed(self):
        t = cc.parse_target("550//arm64")
        self.assertEqual((t.cc, t.arch), (None, "arm64"))

    def test_target_without_driver_is_refused(self):
        with self.assertRaises(cc.Refused):
            cc.parse_target("/7.5")


class TestNodeLabels(unittest.TestCase):
    """--from-node sugar. Pure label logic, so no cluster is needed to test it."""

    GOOD = {"nvidia.com/cuda.driver-version.full": "550.144.03",
            "p6m.dev/cuda.driver-version.major": "550",
            "nvidia.com/gpu.compute.major": "7", "nvidia.com/gpu.compute.minor": "5",
            "kubernetes.io/arch": "amd64"}

    def test_reads_observed_driver_and_compute_capability(self):
        t = cc.target_from_node_labels(self.GOOD, "aks-gpu-0")
        self.assertEqual((t.driver, t.cc, t.arch), ("550.144.03", "7.5", "amd64"))
        self.assertIn("nvidia.com/cuda.driver-version.full", t.provenance)

    def test_contradictory_facts_are_refused(self):
        bad = dict(self.GOOD, **{"p6m.dev/cuda.driver-version.major": "535"})
        with self.assertRaises(cc.Refused) as ctx:
            cc.target_from_node_labels(bad, "aks-gpu-0")
        self.assertIn("contradicts itself", str(ctx.exception))

    def test_node_with_no_driver_fact_is_refused(self):
        with self.assertRaises(cc.Refused) as ctx:
            cc.target_from_node_labels({"kubernetes.io/arch": "amd64"}, "aks-gpu-0")
        self.assertIn("declares no driver fact", str(ctx.exception))

    def test_declared_label_is_used_when_gfd_is_absent(self):
        t = cc.target_from_node_labels({"p6m.dev/cuda.driver-version.major": "550"}, "n")
        self.assertEqual(t.driver, "550")
        self.assertIsNone(t.cc)

    def test_missing_compute_labels_leave_cc_unset_not_guessed(self):
        labels = {k: v for k, v in self.GOOD.items() if "compute" not in k}
        t = cc.target_from_node_labels(labels, "n")
        self.assertIsNone(t.cc)
        self.assertIn("undecided", t.provenance)


class TestDependencyFreedom(unittest.TestCase):
    """cuda-compat.py must stay runnable by someone with nothing but python3.

    This is a product constraint, not a style preference: the customer-facing
    path (YP6M-3479) and the credential-free promise both rest on it, and it is
    the kind of thing a single convenient import quietly ends.
    """

    def test_checker_imports_only_the_standard_library(self):
        import ast
        from helpers import SCRIPT

        with open(SCRIPT) as f:
            tree = ast.parse(f.read())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imported.add(node.module.split(".")[0])
        outside = sorted(imported - set(sys.stdlib_module_names))
        self.assertEqual(outside, [], f"cuda-compat.py imports non-stdlib module(s): {outside}")

    def test_checker_reads_no_credential_environment(self):
        """Registry credentials belong in the action wrapper (YP6M-3480), never here."""
        with open(__import__("helpers").SCRIPT) as f:
            source = f.read()
        for forbidden in ("DOCKER_", "REGISTRY_PASSWORD", "JFROG", "ARTIFACTORY", "os.environ"):
            self.assertNotIn(forbidden, source,
                             f"'{forbidden}' in the credential-free script core")


class TestAggregation(unittest.TestCase):
    """Reducing many verdicts to one. Exit codes are an interface, not a ranking.

    2 is numerically larger than 1, but "could not check this one" is less
    serious than "this one will be refused". Reducing with max() over the exit
    code lets a single unverifiable image mask a definitely-refused one — and
    since exit 2 is a warning by default, the gate then goes green over a
    container the runtime rejects.
    """

    def test_incompatible_outranks_cannot_verify(self):
        self.assertEqual(cc.worse(1, 2), 1)
        self.assertEqual(cc.worse(2, 1), 1)
        self.assertEqual(cc.worst([0, 2, 1]), 1)

    def test_cannot_verify_outranks_compatible(self):
        self.assertEqual(cc.worst([0, 2]), 2)
        self.assertEqual(cc.worst([0, 0]), 0)

    def test_empty_result_set_is_not_a_pass(self):
        self.assertEqual(cc.worst([]), 2)

    def test_a_refused_image_is_not_hidden_by_an_unverifiable_one(self):
        """The end-to-end shape: one INCOMPATIBLE pair plus one CANNOT-VERIFY pair."""
        refused = whole_image("vllm-openai-v0.25.1", "mid=545.23.08/7.5/amd64/Tesla")
        unknown = whole_image("alpine-3.20", T4_POOL)
        self.assertEqual((refused["exit"], unknown["exit"]), (1, 2))
        overall = cc.worst([refused["exit"], unknown["exit"]])
        self.assertEqual(overall, 1, "a definite refusal must survive being mixed with an unknown")
        self.assertEqual(cc.LABELS[overall], "INCOMPATIBLE")

    def test_settled_arch_failure_is_not_downgraded_to_unverified(self):
        """Driver axis undecidable, arch axis definitively wrong -> INCOMPATIBLE."""
        env = {"TORCH_CUDA_ARCH_LIST": "8.0 9.0"}  # no CUDA metadata at all
        worst, _, _ = cc.verdict_for_platform(env, cc.parse_target(T4_POOL))
        self.assertEqual(cc.LABELS[worst], "INCOMPATIBLE")


class TestSeparatorOnlyTerms(unittest.TestCase):
    """A term of nothing but separators, e.g. `cuda>=13.0 , brand=tesla,...`.

    C cannot distinguish "consumed every factor" from "there were no factors":
    both leave and_expr NULL, which breaks the OR scan. Such a term therefore
    decides nothing itself, stops the scan, and carries whatever the previous
    term decided — hiding every term after it. Treating it as vacuously
    satisfied produces a false COMPATIBLE with a confident basis attached to a
    comma. The oracle corpus pins all of these against NVIDIA's own code.
    """

    FACTS = {"cuda": "12.4", "driver": "550.144.03", "arch": "7.5", "brand": "Tesla"}

    def test_separator_term_does_not_satisfy_after_a_failed_term(self):
        res = cc.eval_expression("cuda>=13.0 , brand=tesla,driver>=550,driver<551", self.FACTS)
        self.assertEqual(res["status"], cc.UNSATISFIED)

    def test_separator_term_hides_a_later_satisfying_term(self):
        without = cc.eval_expression("cuda>=99.0 brand=tesla,driver>=550,driver<551", self.FACTS)
        with_sep = cc.eval_expression("cuda>=99.0 ,, brand=tesla,driver>=550,driver<551", self.FACTS)
        self.assertEqual(without["status"], cc.SATISFIED)
        self.assertEqual(with_sep["status"], cc.UNSATISFIED)

    def test_leading_separator_term_carries_the_initial_true(self):
        self.assertEqual(cc.eval_expression(", cuda>=99.0", self.FACTS)["status"], cc.SATISFIED)

    def test_end_to_end_verdict_is_not_a_false_compatible(self):
        env = {"NVIDIA_REQUIRE_CUDA": "cuda>=13.0 , brand=tesla,driver>=550,driver<551"}
        worst, _, basis = cc.verdict_for_platform(env, cc.parse_target(T4_POOL))
        self.assertEqual(cc.LABELS[worst], "INCOMPATIBLE")
        self.assertIsNone(basis)


class TestExitCodes(unittest.TestCase):
    def test_exit_codes_map_to_verdicts(self):
        self.assertEqual(whole_image("vllm-openai-v0.25.1", T4_POOL)["exit"], 0)
        self.assertEqual(whole_image("vllm-openai-v0.25.1",
                                     "mid=545.23.08/7.5/amd64/Tesla")["exit"], 1)
        self.assertEqual(whole_image("alpine-3.20", T4_POOL)["exit"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
