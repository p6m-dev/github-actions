#!/usr/bin/env python3
"""Online half: are the recorded fixtures still faithful to the registry?

The offline suite is hermetic by design, which buys determinism at the cost of
one assumption — that fixtures/*.json still says what the registry says. This
re-fetches each fixture by DIGEST (never by tag, so a tag move cannot reach us)
and compares.

A failure here is not a flake: at a fixed digest the registry cannot legitimately
return different content. It means the recording was wrong, the registry is
serving something else under that digest, or the fetch layer changed shape.

Network absence SKIPS loudly rather than passing quietly — a skipped contract
test is an unverified one, and CI prints the count.

  CUDA_COMPAT_REQUIRE_ONLINE=1  turn "cannot reach the registry" into a failure
"""
import json, os, unittest, urllib.error

from helpers import fixture, load_module

cc = load_module()

FIXTURE_NAMES = ["vllm-openai-v0.25.1", "sglang-v0.5.15", "llamacpp-server-cuda-b10068",
                 "cuda-12.4.1-base", "cuda-12.6.2-base", "alpine-3.20"]
REQUIRE_ONLINE = os.environ.get("CUDA_COMPAT_REQUIRE_ONLINE") == "1"


class TestRegistryContract(unittest.TestCase):
    def _fetch(self, pinned):
        try:
            return cc.fetch_platforms(pinned)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            if REQUIRE_ONLINE:
                self.fail(f"registry unreachable and CUDA_COMPAT_REQUIRE_ONLINE=1: {e}")
            raise unittest.SkipTest(f"registry unreachable — CONTRACT UNVERIFIED for {pinned}: {e}")

    def test_fixtures_match_the_registry(self):
        for name in FIXTURE_NAMES:
            with self.subTest(fixture=name):
                fx = fixture(name)
                self.assertIn("@sha256:", fx["pinned"])
                _, live = self._fetch(fx["pinned"])

                recorded = {p["platform"]: p for p in fx["platforms"]}
                fetched = {p["platform"]: p for p in live}
                self.assertEqual(sorted(recorded), sorted(fetched),
                                 f"{name}: platform set drifted from the recording")

                for platform_name, rec in recorded.items():
                    got = fetched[platform_name]
                    self.assertEqual(rec["digest"], got["digest"],
                                     f"{name} {platform_name}: child digest changed under a pinned index")
                    for key, value in rec["env"].items():
                        self.assertEqual(value, got["env"].get(key),
                                         f"{name} {platform_name}: {key} differs from the recording")

    def test_verdict_holds_end_to_end_against_the_live_registry(self):
        """The headline claim, re-checked through the real fetch path rather than a recording."""
        fx = fixture("vllm-openai-v0.25.1")
        _, live = self._fetch(fx["pinned"])
        result = cc.evaluate(fx["pinned"], live, fx["index_digest"],
                             cc.parse_target("t4=550.144.03/7.5/amd64/Tesla,Nvidia"))
        self.assertEqual(result["verdict"], "COMPATIBLE")
        self.assertIn("forward compatibility", result["basis"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
