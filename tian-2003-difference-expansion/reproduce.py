"""Run the full experiment twice and compare exact numeric records."""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from experiment_support import compare_results, run_script


def main() -> None:
    for name in ('run-a', 'run-b'):
        destination = ROOT / "outputs" / name
        if destination.exists():
            shutil.rmtree(destination)
        run_script(ROOT, ["run.py", name])
    run_script(ROOT, ["verify.py"])
    count = compare_results(ROOT / "results", ROOT / "outputs")
    print(f"PASS tian-2003-difference-expansion: full run twice; {count} recorded files identical")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, OSError, subprocess.CalledProcessError) as error:
        print(f"FAIL tian-2003-difference-expansion: {error}", file=sys.stderr)
        raise SystemExit(1) from error
