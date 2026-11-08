from pathlib import Path
import verify

verify.main(Path(__file__).resolve().parent / "reproduced")
