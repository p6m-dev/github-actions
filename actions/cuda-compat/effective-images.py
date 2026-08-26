#!/usr/bin/env python3
"""Resolve the images an Installable will ACTUALLY deploy, defaults included.

An Installable's declared values are not the deployed image set. llmkube pins
`targetRevision: 0.9.*` with an empty runtimeImages block, so the images that
reach a node are compiled into whichever operator build that floating range
resolves to today. When those built-in defaults move, nothing changes in git:
no diff, no PR, no review surface. That is why the scheduled run — not the
pull_request trigger — is the primary one.

This resolves, in order:
  1. the Installable's targetRevision constraint -> a concrete chart version
     (from the registry's tag list, i.e. what ArgoCD would pick today)
  2. that chart's OCI config blob -> its exact appVersion
  3. the operator source at that tag -> the built-in default runtime images
  4. the Installable's own runtimeImages -> overrides layered on top

Every step refuses loudly rather than guessing. A partial answer here would be
worse than none: it would report a clean verdict over an image set that is not
the deployed one.

  ./effective-images.py --installable path/to/llmkube.yaml [--json] [--github-output]

Needs PyYAML (CI-side tool). Set GITHUB_TOKEN to avoid anonymous rate limits.
"""
import argparse, base64, fnmatch, json, os, re, sys, urllib.error, urllib.request

try:
    import yaml
except ImportError:
    sys.exit("effective-images.py needs PyYAML (pip install pyyaml).")

HERE = os.path.dirname(os.path.abspath(__file__))
OPERATOR_TABLE = os.path.join(HERE, "tables", "operator-defaults.yaml")

sys.path.insert(0, HERE)
import importlib.util

_spec = importlib.util.spec_from_file_location("cuda_compat", os.path.join(HERE, "cuda-compat.py"))
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)

# A Go string literal that is shaped like an image reference.
IMAGE_LITERAL = re.compile(
    r'"((?:[a-z0-9][a-z0-9.\-]*(?::\d+)?/)?[a-z0-9][a-z0-9._\-]*(?:/[a-z0-9][a-z0-9._\-]*)*'
    r'(?::[\w][\w.\-]*|@sha256:[a-f0-9]{64}))"')
CONST_NAME = re.compile(r"^\s*(?:const\s+)?(\w+)\s*=|func\s+\([^)]*\)\s*(\w+)\s*\(")


class Refused(Exception):
    """Refuse rather than report a partial image set as if it were complete."""


def gh_get(path):
    url = f"https://api.github.com/{path}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        return json.loads(urllib.request.urlopen(req, timeout=30).read())
    except urllib.error.HTTPError as e:
        hint = " (set GITHUB_TOKEN — anonymous API access is rate limited)" if e.code == 403 else ""
        raise Refused(f"GitHub API {e.code} for {path}{hint}")
    except urllib.error.URLError as e:
        raise Refused(f"GitHub API unreachable for {path}: {e}")


def resolve_version(registry, repo, constraint):
    """Pick the version ArgoCD would pick today. Only patterns we can honour."""
    if re.fullmatch(r"\d+(\.\d+)*", constraint):
        return constraint, f"exact pin {constraint}"
    if not re.fullmatch(r"[\d.]*\*", constraint):
        raise Refused(
            f"targetRevision '{constraint}' is a constraint form this resolver does not "
            f"implement (only exact pins and trailing-* globs). Refusing to guess which "
            f"version is deployed.")
    reg = cc.Registry(registry, repo)
    body, _ = reg._get(f"https://{registry}/v2/{repo}/tags/list?n=1000", "application/json")
    tags = json.loads(body).get("tags") or []
    matched = [t for t in tags if fnmatch.fnmatch(t, constraint) and re.fullmatch(r"[\d.]+", t)]
    if not matched:
        raise Refused(f"no tag in {registry}/{repo} matches '{constraint}'")
    newest = max(matched, key=lambda t: [int(p) for p in t.split(".")])
    return newest, (f"'{constraint}' resolved against {len(tags)} tags -> {newest} "
                    f"(floating: this can change with no diff anywhere in git)")


def chart_app_version(registry, repo, version):
    reg = cc.Registry(registry, repo)
    manifest, _ = reg.manifest(version)
    config_type = manifest.get("config", {}).get("mediaType", "")
    if "helm" not in config_type:
        raise Refused(f"{registry}/{repo}:{version} is not a Helm chart artifact ({config_type})")
    meta = reg.blob(manifest["config"]["digest"])
    app = meta.get("appVersion")
    if not app:
        raise Refused(f"chart {version} declares no appVersion — cannot locate the operator source")
    return app, meta


def builtin_defaults(spec, app_version):
    """Read the operator's compiled-in image constants at its source tag."""
    tag = spec["tag_format"].format(appVersion=app_version)
    repo = spec["source_repo"]
    listing = gh_get(f"repos/{repo}/contents/{spec['source_dir']}?ref={tag}")
    names = [f["name"] for f in listing if fnmatch.fnmatch(f["name"], spec["file_pattern"])
             and not f["name"].endswith("_test.go")]
    if not names:
        raise Refused(f"no {spec['file_pattern']} under {repo}/{spec['source_dir']} at {tag}")

    found = []
    for name in sorted(names):
        blob = gh_get(f"repos/{repo}/contents/{spec['source_dir']}/{name}?ref={tag}")
        source = base64.b64decode(blob["content"]).decode("utf-8", "replace")
        for line in source.splitlines():
            if line.lstrip().startswith("//"):
                continue
            for image in IMAGE_LITERAL.findall(line):
                if "/" not in image and "@" not in image:
                    continue  # a bare word:version, not an image ref
                match = CONST_NAME.match(line)
                symbol = next((g for g in match.groups() if g), None) if match else None
                found.append({"image": image, "file": name, "symbol": symbol or "(inline)"})

    unique = {}
    for entry in found:
        unique.setdefault(entry["image"], entry)
    images = list(unique.values())
    if len(images) < spec.get("expect_at_least", 1):
        raise Refused(
            f"found only {len(images)} default image(s) in {repo}@{tag}, expected at least "
            f"{spec['expect_at_least']} — the source shape changed, so extraction can no longer "
            f"be trusted to be complete. Update tables/operator-defaults.yaml deliberately.")
    return tag, images


def classify(images, markers):
    """Split NVIDIA-relevant images from the rest. Nothing is dropped silently."""
    nvidia, other = [], []
    for entry in images:
        haystack = f"{entry['image']} {entry['symbol']}".lower()
        hit = next((m for m in markers if m in haystack), None)
        if hit:
            other.append(dict(entry, excluded_because=f"non-NVIDIA backend (matched '{hit}')"))
        else:
            nvidia.append(entry)
    return nvidia, other


def installable_source(path):
    with open(path) as f:
        try:
            doc = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise Refused(f"{path} is not parseable YAML: {e}")
    source = ((doc or {}).get("spec") or {}).get("source") or {}
    for field in ("repoURL", "chart", "targetRevision"):
        if not source.get(field):
            raise Refused(f"{path} declares no spec.source.{field}")
    template = source.get("template") or ""
    overrides = {}
    if template:
        if "{{" in template:
            # Handlebars in the values means the deployed values depend on
            # cluster context this resolver does not have.
            raise Refused(f"{path} template contains handlebars; the effective values differ "
                          f"per cluster and cannot be resolved here")
        try:
            values = yaml.safe_load(template) or {}
        except yaml.YAMLError as e:
            raise Refused(f"{path} template is not parseable YAML: {e}")
        overrides = values.get("runtimeImages") or {}
    return source, overrides


def resolve(installable_path):
    source, overrides = installable_source(installable_path)
    registry, chart = source["repoURL"].rstrip("/"), source["chart"]
    key = f"{registry}/{chart}"
    table = yaml.safe_load(open(OPERATOR_TABLE))["charts"]
    spec = table.get(key)
    if not spec:
        raise Refused(f"chart '{key}' is not in tables/operator-defaults.yaml — its built-in "
                      f"defaults cannot be located, so the effective image set is unknown")

    version, how = resolve_version(registry, chart, str(source["targetRevision"]))
    app_version, _ = chart_app_version(registry, chart, version)
    tag, defaults = builtin_defaults(spec, app_version)
    nvidia, other = classify(defaults, spec.get("non_nvidia_markers", []))

    images = [dict(e, origin="operator built-in default") for e in nvidia]
    for backend, image in sorted(overrides.items()):
        if not isinstance(image, str):
            continue
        images = [i for i in images if backend.lower() not in i["symbol"].lower()]
        images.append({"image": image, "file": os.path.basename(installable_path),
                       "symbol": f"{spec['override_path']}.{backend}",
                       "origin": "Installable override"})

    return {
        "installable": os.path.basename(installable_path),
        "chart": key,
        "constraint": str(source["targetRevision"]),
        "resolved_version": version,
        "resolution": how,
        "app_version": app_version,
        "operator_tag": tag,
        "images": images,
        "excluded": other,
        "overrides_declared": sorted(overrides),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--installable", required=True)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--github-output", action="store_true")
    args = ap.parse_args()

    try:
        result = resolve(args.installable)
    except Refused as e:
        print(f"CANNOT-VERIFY: {e}", file=sys.stderr)
        if args.github_output and os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                f.write("images<<CC_EOF\n\nCC_EOF\n")
                f.write(f"refusal={e}\n".replace("\n", " ") + "\n")
        return 2

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"{result['installable']}: {result['chart']}")
        print(f"  targetRevision {result['constraint']} -> chart {result['resolved_version']}, "
              f"operator {result['operator_tag']}")
        print(f"  {result['resolution']}")
        if not result["overrides_declared"]:
            print("  runtimeImages: none declared — every image below is an operator built-in "
                  "default, which is what makes the floating pin a review-free change path")
        print("\n  effective NVIDIA image set:")
        for entry in result["images"]:
            print(f"    {entry['image']}\n        {entry['origin']} ({entry['file']}: {entry['symbol']})")
        if result["excluded"]:
            print("\n  listed but not sent for a CUDA verdict:")
            for entry in result["excluded"]:
                print(f"    {entry['image']} — {entry['excluded_because']}")

    if args.github_output and os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write("images<<CC_EOF\n" + "\n".join(e["image"] for e in result["images"]) + "\nCC_EOF\n")
            f.write(f"chart-version={result['resolved_version']}\n")
            f.write(f"operator-tag={result['operator_tag']}\n")
            f.write("refusal=\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
