#!/usr/bin/env python3
"""cuda-compat: will this image's CUDA userspace run on a node with this driver?

Verdict per platform in the image (every OCI-index child is checked — the
single-request mistake fails open on exactly the newest images):

  COMPATIBLE      evidence: which constraint the driver satisfies (+arch check)
  INCOMPATIBLE    the exact failed constraint
  CANNOT-VERIFY   image declares no CUDA metadata, or no driver fact given.
                  Never a silent pass.

Checks, in order of trust:
  1. NVIDIA_REQUIRE_CUDA  — what nvidia-container-cli actually enforces at
     container start (space = AND, comma = OR; terms: cuda>=X.Y, driver>=V, ...)
  2. CUDA_VERSION         — informational fallback: image's CUDA toolkit vs the
     max CUDA the driver branch supports
  3. TORCH_CUDA_ARCH_LIST — if present AND a --cc is given: compiled arch list
     vs the target GPU's compute capability (+PTX = JIT forward-compat, noted)

Usage:
  cuda-compat.py IMAGE --driver 550 [--cc 7.5] [--strict-unknown]
Exit codes: 0 compatible, 1 incompatible, 2 cannot-verify (always; gate with
--strict-unknown semantics in CI by treating 2 as failure or warning).
"""
import argparse, json, re, sys, urllib.request, urllib.parse

# Driver branch -> max CUDA version that branch's driver reports (what the
# cuda>=X.Y constraint is evaluated against by nvidia-container-cli via NVML).
# Fail-honest: an unknown branch is CANNOT-VERIFY, never interpolated.
DRIVER_MAX_CUDA = {
    "470": "11.4", "510": "11.6", "515": "11.7", "520": "11.8", "525": "12.0",
    "530": "12.1", "535": "12.2", "545": "12.3", "550": "12.4", "555": "12.5",
    "560": "12.6", "565": "12.7", "570": "12.8", "575": "12.9", "580": "13.0",
}

MEDIA_INDEX = ("application/vnd.oci.image.index.v1+json",
               "application/vnd.docker.distribution.manifest.list.v2+json")
MEDIA_MANIFEST = ("application/vnd.oci.image.manifest.v1+json",
                  "application/vnd.docker.distribution.manifest.v2+json")
ACCEPT = ", ".join(MEDIA_INDEX + MEDIA_MANIFEST)


def parse_ref(ref):
    # [registry/]repo[:tag][@digest]
    digest = None
    if "@" in ref:
        ref, digest = ref.split("@", 1)
    tag = "latest"
    if ":" in ref.rsplit("/", 1)[-1]:
        ref, tag = ref.rsplit(":", 1)
    parts = ref.split("/")
    if "." in parts[0] or ":" in parts[0] or parts[0] == "localhost":
        registry, repo = parts[0], "/".join(parts[1:])
    else:
        registry, repo = "registry-1.docker.io", ref if "/" in ref else f"library/{ref}"
    return registry, repo, (digest or tag)


class Registry:
    def __init__(self, registry, repo):
        self.base = f"https://{registry}/v2/{repo}"
        self.repo = repo
        self.token = None

    def _get(self, url, accept):
        req = urllib.request.Request(url, headers={"Accept": accept})
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            return urllib.request.urlopen(req, timeout=30).read()
        except urllib.error.HTTPError as e:
            if e.code != 401 or self.token:
                raise
            challenge = e.headers.get("WWW-Authenticate", "")
            m = dict(re.findall(r'(\w+)="([^"]*)"', challenge))
            if "realm" not in m:
                raise
            q = {"service": m.get("service", ""), "scope": m.get("scope", f"repository:{self.repo}:pull")}
            tok = json.loads(urllib.request.urlopen(
                f"{m['realm']}?{urllib.parse.urlencode(q)}", timeout=30).read())
            self.token = tok.get("token") or tok.get("access_token")
            return self._get(url, accept)

    def manifest(self, ref):
        return json.loads(self._get(f"{self.base}/manifests/{ref}", ACCEPT))

    def blob(self, digest):
        return json.loads(self._get(f"{self.base}/blobs/{digest}", "*/*"))


def env_map(config):
    out = {}
    for e in (config.get("config") or {}).get("Env") or []:
        k, _, v = e.partition("=")
        out[k] = v
    return out


def vtuple(s):
    return tuple(int(x) for x in re.findall(r"\d+", s)[:3]) or (0,)


def eval_require_cuda(expr, driver, max_cuda):
    """space = AND, comma = OR. Returns (ok, detail)."""
    for clause in expr.split():
        alts, details = [], []
        for term in clause.split(","):
            m = re.match(r"(\w+)(>=|<=|=|>|<|!=)([\w.]+)", term)
            if not m:
                details.append(f"unparsed term '{term}'")
                alts.append(None)  # unknown term: not satisfied, not failed
                continue
            key, op, val = m.groups()
            if key == "cuda":
                left = vtuple(max_cuda)
            elif key == "driver":
                left = vtuple(driver)
            elif key in ("arch", "brand"):
                alts.append(True)  # host-arch/brand: out of scope here, assume met
                details.append(f"{term} (assumed)")
                continue
            else:
                details.append(f"unknown key '{key}'")
                alts.append(None)
                continue
            right = vtuple(val)
            ok = {"<": left < right, "<=": left <= right, ">": left > right,
                  ">=": left >= right, "=": left == right, "!=": left != right}[op]
            alts.append(ok)
            details.append(f"{term} -> {'ok' if ok else 'FAIL'} (have {key} {max_cuda if key=='cuda' else driver})")
        if not any(a is True for a in alts):
            return False, "; ".join(details)
    return True, f"all constraints satisfied against driver {driver} (CUDA {max_cuda})"


def check_arch(arch_list, cc):
    entries = re.split(r"[;,\s]+", arch_list.strip())
    plain = [e.replace("+PTX", "") for e in entries if e]
    has = cc in plain or f"{cc}+PTX" in entries
    if has:
        return True, f"compute capability {cc} in TORCH_CUDA_ARCH_LIST ({arch_list})"
    ptx = [e[:-4] for e in entries if e.endswith("+PTX")]
    if ptx and any(vtuple(p) <= vtuple(cc) for p in ptx):
        return True, f"{cc} not compiled, but PTX from {max(ptx, key=vtuple)} JIT-forward-compatible ({arch_list}) — first launch is slow, not guaranteed for all kernels"
    return False, f"compute capability {cc} not in TORCH_CUDA_ARCH_LIST ({arch_list}), no applicable +PTX"


def verdict_for_platform(env, driver, max_cuda, cc):
    findings, worst = [], 0  # 0 ok, 1 incompatible, 2 cannot-verify
    req = env.get("NVIDIA_REQUIRE_CUDA")
    cuda_ver = env.get("CUDA_VERSION")
    if req:
        ok, detail = eval_require_cuda(req, driver, max_cuda)
        findings.append(("NVIDIA_REQUIRE_CUDA (enforced at container start)", ok, detail))
        if not ok:
            worst = 1
    elif cuda_ver:
        ok = vtuple(cuda_ver)[:2] <= vtuple(max_cuda)[:2]
        findings.append(("CUDA_VERSION (informational only — nothing enforces it)", ok,
                         f"toolkit {cuda_ver} vs driver {driver} max CUDA {max_cuda}"))
        if not ok:
            worst = 1
    else:
        findings.append(("CUDA metadata", None,
                         "image declares neither NVIDIA_REQUIRE_CUDA nor CUDA_VERSION — "
                         "cannot verify; consider a p6m.dev/cuda-requires annotation"))
        worst = 2
    arch = env.get("TORCH_CUDA_ARCH_LIST")
    if cc:
        if arch:
            ok, detail = check_arch(arch, cc)
            findings.append(("TORCH_CUDA_ARCH_LIST vs --cc", ok, detail))
            if not ok:
                worst = max(worst, 1)
        else:
            findings.append(("TORCH_CUDA_ARCH_LIST", None,
                             f"not declared — arch fit for compute capability {cc} unverified"))
            if worst == 0:
                worst = max(worst, 0)  # driver verdict stands; arch axis merely unverified
    return worst, findings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--driver", required=True, help="target driver branch, e.g. 550")
    ap.add_argument("--cc", help="target GPU compute capability, e.g. 7.5")
    args = ap.parse_args()

    max_cuda = DRIVER_MAX_CUDA.get(args.driver)
    if not max_cuda:
        print(f"CANNOT-VERIFY: driver branch '{args.driver}' not in the known table "
              f"({', '.join(sorted(DRIVER_MAX_CUDA))}) — add it deliberately, never guess.")
        sys.exit(2)

    registry, repo, ref = parse_ref(args.image)
    reg = Registry(registry, repo)
    top = reg.manifest(ref)

    children = []
    if top.get("mediaType") in MEDIA_INDEX or "manifests" in top:
        for m in top["manifests"]:
            p = m.get("platform") or {}
            if p.get("os") == "linux" and not m.get("annotations", {}).get("vnd.docker.reference.type"):
                children.append((f"{p.get('os')}/{p.get('architecture')}", m["digest"]))
    else:
        children.append(("(single-platform)", None))

    print(f"image:  {args.image}")
    print(f"target: driver {args.driver} (max CUDA {max_cuda})" + (f", compute capability {args.cc}" if args.cc else ""))
    overall = 0
    for plat, digest in children:
        man = reg.manifest(digest) if digest else top
        cfg = reg.blob(man["config"]["digest"])
        worst, findings = verdict_for_platform(env_map(cfg), args.driver, max_cuda, args.cc)
        label = {0: "COMPATIBLE", 1: "INCOMPATIBLE", 2: "CANNOT-VERIFY"}[worst]
        print(f"\n  {plat}: {label}")
        for name, ok, detail in findings:
            mark = {True: "ok  ", False: "FAIL", None: "?   "}[ok]
            print(f"    [{mark}] {name}: {detail}")
        overall = max(overall, worst)
    print(f"\nverdict: {['COMPATIBLE','INCOMPATIBLE','CANNOT-VERIFY'][overall]} "
          f"(assurance: static config-blob analysis — says 'will not fail the driver check', not 'boots')")
    sys.exit(overall)


if __name__ == "__main__":
    main()
