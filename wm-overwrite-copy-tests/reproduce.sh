#!/usr/bin/env bash
# Runs the four scenarios in turn. Needs an NVIDIA GPU and COCO val2017 at ./coco/val2017.
# Each scenario writes outputs/<scenario>/results.json and per_image.csv.
set -euo pipefail
cd "$(dirname "$0")"
for s in baseline overwrite_same cross_method copy_attack; do
  sed -e "s/^scenario: .*/scenario: $s/" -e "s#^output_dir: .*#output_dir: outputs/$s/#" config.yaml > "config.$s.yaml"
  bash run.sh "config.$s.yaml"
done
