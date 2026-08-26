#!/usr/bin/env python3
"""Render a cuda-compat JSON report: human text, markdown table, or the verdict.

Kept separate from cuda-compat.py so the action runs the checker exactly once —
each invocation is a round of registry requests, and the scheduled sweep covers
the whole estate.

  report.py --text REPORT       what the checker would have printed
  report.py --markdown REPORT   table for an issue body / PR comment / job summary
  report.py --verdict REPORT    the single worst verdict
"""
import argparse, importlib.util, json, os, sys

ICON = {"COMPATIBLE": "✅", "INCOMPATIBLE": "❌", "CANNOT-VERIFY": "⚠️"}


def load_checker():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda-compat.py")
    spec = importlib.util.spec_from_file_location("cuda_compat", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def short(ref, width=52):
    """Keep a digest-pinned ref readable without losing which digest it was."""
    if "@sha256:" in ref:
        repo, digest = ref.split("@sha256:")
        return f"{repo}@{digest[:12]}"
    return ref if len(ref) <= width else "…" + ref[-(width - 1):]


def compact_basis(basis):
    """The full basis is a paragraph. A table cell needs the identifying half."""
    if basis and basis.startswith("driver-branch allowance"):
        term = basis.split("'")[1] if "'" in basis else "a driver-branch term"
        return f"forward-compat via `{term}` (needs the image's cuda-compat runtime; datacenter GPUs only)"
    return basis


def reason_for(result):
    """Why this verdict — for every verdict, not only failures.

    A CANNOT-VERIFY with an empty reason column is the worst cell in the table:
    it looks like the tool had nothing to say, when in fact it has a specific
    thing to say and that thing is the whole point of the verdict.
    """
    if result.get("refusal"):
        return result["refusal"]
    if result["verdict"] == "COMPATIBLE":
        # A COMPATIBLE that rests on an unmeasured axis must say so here. The
        # markdown table is the whole artifact for anyone reading the PR comment
        # or the findings issue, and "the driver check passes" is not the same
        # claim as "this will work" when kernel fit was never verified.
        base = compact_basis(result.get("basis") or "")
        caveats = [f["detail"] for p in result["platforms"] if p["targeted"]
                   for f in p["findings"] if f["status"] == "unknown"]
        return base + (" — " + "; ".join(caveats) if caveats else "")
    wanted = "fail" if result["verdict"] == "INCOMPATIBLE" else "unknown"
    for p in result["platforms"]:
        if p["targeted"]:
            hits = [f["detail"] for f in p["findings"] if f["status"] == wanted]
            if hits:
                return hits[0]
    return result.get("basis") or ""


def markdown(report):
    lines = [f"### cuda-compat: {ICON.get(report['verdict'], '')} **{report['verdict']}**", ""]
    lines += ["| image | target | verdict | basis / reason |", "|---|---|---|---|"]
    for r in report["results"]:
        target = r["target"]
        basis = reason_for(r).replace("|", "\\|").replace("\n", " ")
        if len(basis) > 220:
            basis = basis[:217] + "…"
        lines.append(f"| `{short(r['image'])}` | `{target['label']}` | "
                     f"{ICON.get(r['verdict'], '')} {r['verdict']} | {basis} |")
    lines += ["", f"_Assurance: {report['assurance']}_"]

    digests = {r["image"]: r["digest"] for r in report["results"] if r.get("digest")}
    if digests:
        lines += ["", "<details><summary>Digests verdicted</summary>", ""]
        lines += [f"- `{image}` → `{digest}`" for image, digest in sorted(digests.items())]
        lines += ["", "</details>"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--text", action="store_true")
    mode.add_argument("--markdown", action="store_true")
    mode.add_argument("--verdict", action="store_true")
    args = ap.parse_args()

    with open(args.report) as f:
        report = json.load(f)

    if args.verdict:
        print(report["verdict"])
    elif args.markdown:
        print(markdown(report))
    else:
        load_checker().print_text(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
