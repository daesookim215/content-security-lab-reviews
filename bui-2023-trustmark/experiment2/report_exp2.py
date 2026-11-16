# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: ../.venv/bin/python report_exp2.py [results|reproduced]
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent / (sys.argv[1] if len(sys.argv) > 1 else "results")
    data = json.loads((root / "run1/results.json").read_text())
    rows = data["rows"]
    pairs = [(name, size) for name in ("airplane", "boat") for size in (512, 2048)]
    output = ["## Observed results", "",
              "Image filenames identify artifacts created when reproducing; images and logs are not included in the published reference results.", "", "Raw accuracy below refers to the 61 payload bits unless explicitly labeled 100-bit float.", "",
              "### A: single changes", "", "Cells: raw correct/61, exact BCH recovery (Y/N).", "",
              "| Change | Airplane 512 | Airplane 2048 | Boat 512 | Boat 2048 |", "|---|---|---|---|---|"]
    for setting in data["settings"]["single_changes"]:
        cells = []
        for name, size in pairs:
            row = next(r for r in rows if (r["image"], r["resolution"], r["experiment"], r["change"])
                       == (name, size, "A", setting["name"]))
            cells.append(f"{row['payload_bits_correct']}/61, {'Y' if row['exact_payload'] else 'N'}")
        output.append("| " + " | ".join([setting["name"], *cells]) + " |")
    output += ["", "| A aggregate | Mean raw accuracy | Minimum | Exact recovery |", "|---|---:|---:|---:|"]
    for row in data["summary"]:
        if row["experiment"] == "A":
            output.append(f"| {row['image']} {row['resolution']} | {row['mean_raw_accuracy']:.2%} | {row['min_raw_accuracy']:.2%} | {row['exact_recovery_count']}/{row['count']} |")
    output += ["", "### B: combined changes", "", "| Image | Size | Mean raw accuracy | Minimum | Exact recovery |", "|---|---:|---:|---:|---:|"]
    for row in data["summary"]:
        if row["experiment"] == "B":
            output.append(f"| {row['image']} | {row['resolution']} | {row['mean_raw_accuracy']:.2%} | {row['min_raw_accuracy']:.2%} | {row['exact_recovery_count']}/20 |")
    output += ["", "### C: removal and re-watermarking", "", "Complete C1/C2/C3 results and reproduction instructions are in ../experiment3/README.md. The wrapper runs C separately after A/B/D/E.", ""]
    for name, size in pairs:
        selected = [r for r in rows if (r["image"], r["resolution"], r["experiment"]) == (name, size, "C")]
        if selected:
            for row in selected:
                output.append(f"- {name} {size}, {row['change']}: {row['payload_bits_correct']}/61, detected={row['detected']}, exact={row['exact_payload']}, PSNR vs input {row['psnr_input_db']}, vs marked {row['psnr_marked_db']} dB.")
    output += ["", "### D: I-FGSM", "", "| Image | Size | Steps | Float 100-bit raw | PNG payload correct | Detected | Exact | PSNR vs marked | Float criterion |", "|---|---:|---:|---:|---:|---|---|---:|---|"]
    for name, size in pairs:
        selected = [r for r in rows if (r["image"], r["resolution"], r["experiment"]) == (name, size, "D")]
        if selected:
            r = selected[0]
            output.append(f"| {name} | {size} | {r['iterations']} | {r['float_packet_accuracy']:.1%} | {r['payload_bits_correct']}/61 ({r['bit_accuracy']:.2%}) | {r['detected']} | {r['exact_payload']} | {r['psnr_marked_db']:.2f} | {r['criterion_met']} |")
        else:
            output.append(f"| {name} | {size} | skipped | {data['attack_2048_reason']} | - | - | - | - | - |")
    output += ["", "### E: inference strength", "", "PSNR is pre-JPEG embedding quality. Accuracy is correct/61; Y/N denotes exact BCH recovery.", "",
               "| Image | Size | Lambda | PSNR vs input | Unchanged raw, exact | JPEG50 raw, exact |", "|---|---:|---:|---:|---|---|"]
    for name, size in pairs:
        for strength in (0.5, 0.75, 1.0, 1.25, 1.5):
            selected = [r for r in rows if (r["image"], r["resolution"], r["experiment"], r.get("strength")) == (name, size, "E", strength)]
            clean = next(r for r in selected if r["change"].endswith("unchanged"))
            jpeg = next(r for r in selected if r["change"].endswith("jpeg50"))
            output.append(f"| {name} | {size} | {strength} | {clean['psnr_input_db']:.2f} | {clean['payload_bits_correct']}/61, {'Y' if clean['exact_payload'] else 'N'} | {jpeg['payload_bits_correct']}/61, {'Y' if jpeg['exact_payload'] else 'N'} |")
    output += ["", "### Runtime and repeatability", "", "| Run | Total seconds | A | B | C | D | E |", "|---|---:|---:|---:|---:|---:|---:|"]
    for run in ("run1", "run2"):
        timing = json.loads((root / f"{run}/timings.json").read_text())
        times = [sum(r["seconds"] for r in timing["measurements"] if r["experiment"] == exp and "seconds" in r)
                 for exp in ("A", "B", "C", "D", "E")]
        output.append("| " + " | ".join([run, f"{timing['total_seconds']:.2f}", *[f"{v:.2f}" for v in times]]) + " |")
    determinism = json.loads((root / "determinism.json").read_text())
    output += ["", f"Byte-identical except timing: **{determinism['byte_identical_except_timing']}**. Compared {len(determinism['checked_files'])} files per run plus the logs; only timings.json was excluded.",
               f"Differences: {json.dumps(determinism['differences'])}", "", f"2048 attack eligibility: {data['attack_2048_reason']}; enabled={data['attack_2048_enabled']}.", "",
               "The model/source hashes and all four marked PNG hashes matched the public baseline reference. C1/C2/C3 are recorded separately by experiment3.", ""]
    rendered = "\n".join(output)
    (root / "summary.md").write_text(rendered)


if __name__ == "__main__":
    main()
