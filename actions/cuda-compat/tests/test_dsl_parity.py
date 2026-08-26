#!/usr/bin/env python3
"""Differential test: cuda-compat.py's DSL port vs libnvidia-container's own code.

Every case in oracle/golden.json was decided by compiling NVIDIA's src/cli/dsl.c
verbatim (see oracle/README.md). This test replays them through the Python port.
A failure here means our verdicts and the runtime's behaviour have diverged —
which is the one bug class this tool must never have, because it is the bug class
that reads as competence right up until a container refuses to start.

Hermetic: no compiler, no network, no third-party source at test time.
"""
import json, os, unittest

from helpers import load_module

cc = load_module()

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "oracle", "golden.json")


def python_verdict(expr, facts):
    """Run the port and reduce it to the oracle's three-state vocabulary."""
    try:
        res = cc.eval_expression(expr, facts)
    except cc.Refused:
        return cc.MALFORMED
    return res["status"]


class TestDSLParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(GOLDEN) as f:
            cls.doc = json.load(f)
        cls.cases = cls.doc["cases"]

    def test_corpus_is_present_and_discriminating(self):
        """A corpus that only ever says one thing would pass any port."""
        self.assertGreaterEqual(len(self.cases), 300)
        seen = {c["expect"] for c in self.cases}
        self.assertEqual(seen, {"SATISFIED", "UNSATISFIED", "MALFORMED"},
                         "corpus lost a verdict class — it can no longer catch an inverted operator")
        for expect in seen:
            self.assertGreaterEqual(sum(c["expect"] == expect for c in self.cases), 20,
                                    f"too few {expect} cases to be meaningful")

    def test_provenance_recorded(self):
        self.assertEqual(len(self.doc.get("dsl_c_sha256", "")), 64,
                         "golden table must record which dsl.c produced it")
        self.assertEqual(len(self.doc.get("upstream_commit", "")), 40)

    def test_port_matches_nvidia_on_every_case(self):
        mismatches = []
        for c in self.cases:
            facts = {"cuda": c["cuda"], "driver": c["driver"],
                     "arch": c["arch"], "brand": c["brand"]}
            got = python_verdict(c["expr"], facts)
            if got != c["expect"]:
                mismatches.append(
                    f"\n  {c['source']}\n  expr:   {c['expr'][:120]}\n"
                    f"  facts:  driver={c['driver']} cuda={c['cuda']} arch={c['arch']} brand={c['brand']}\n"
                    f"  nvidia: {c['expect']}\n  ours:   {got}")
        self.assertEqual(mismatches, [], f"{len(mismatches)} divergence(s) from NVIDIA:"
                                         + "".join(mismatches[:10]))

    def test_real_image_expressions_are_in_the_corpus(self):
        """Synthetic cases alone would miss the AND/OR inversion this tool was born from."""
        sources = {c["source"].split(":")[0] for c in self.cases}
        self.assertIn("vllm-openai-v0.25.1", sources)
        real = [c for c in self.cases if c["source"] != "synthetic"]
        self.assertGreaterEqual(len(real), 50)
        # The inversion's signature: the vllm string must be SATISFIED at 550/Tesla
        # and UNSATISFIED at 545/Tesla. A port that reads space as AND fails both.
        vllm = {(c["driver"], c["brand"]): c["expect"] for c in real
                if c["source"].startswith("vllm-openai-v0.25.1:linux/amd64")}
        self.assertEqual(vllm[("550.144.03", "Tesla")], "SATISFIED")
        self.assertEqual(vllm[("545.23.08", "Tesla")], "UNSATISFIED")
        # Forward compatibility is offered to datacenter GPUs only: same driver,
        # consumer brand, opposite answer. This pair is the tool's whole thesis.
        self.assertEqual(vllm[("550.144.03", "GeForce")], "UNSATISFIED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
