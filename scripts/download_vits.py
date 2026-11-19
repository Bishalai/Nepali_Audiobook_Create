"""Download the Nepali VITS model into <project>/models/vits (no HF caches).

Run:  python scripts\download_vits.py
Files land directly at their final paths via streaming HTTPS - HuggingFace
hub caches (HF_HOME etc.) are never created.
"""

import pathlib
import sys

import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEST = ROOT / "models" / "vits"

FILES = [
    # authors recommend G_100000.pth (clearest checkpoint)
    ("https://huggingface.co/Dragneel/nepali-vits-tts/resolve/main/G_100000.pth",
     "G_100000.pth"),
    ("https://huggingface.co/Ichigo369/nepali-vits-tts/resolve/main/nepali_base.json",
     "nepali_base.json"),
    ("https://huggingface.co/Ichigo369/nepali-vits-tts/resolve/main/nepali_symbols.py",
     "nepali_symbols.py"),
    ("https://huggingface.co/Ichigo369/nepali-vits-tts/resolve/main/nepali_cleaners.py",
     "nepali_cleaners.py"),
]

VITS_REPO = "https://github.com/jaywalnut310/vits"


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    for url, name in FILES:
        path = DEST / name
        if path.exists() and path.stat().st_size > 1000:
            print(f"skip (exists): {name}")
            continue
        print(f"downloading {name} ...", flush=True)
        with requests.get(url, stream=True, timeout=180) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            done = 0
            with open(path, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if total and done % (50 << 20) < (1 << 20):
                        print(f"  {done / 1e6:.0f}/{total / 1e6:.0f} MB", flush=True)
        print(f"  saved: models/vits/{name} ({path.stat().st_size:,} bytes)")

    vits_dir = ROOT / "third_party" / "vits"
    if not vits_dir.is_dir():
        import subprocess
        print("cloning VITS code ...")
        subprocess.run(["git", "clone", "--depth", "1", VITS_REPO, str(vits_dir)],
                       check=True)

    # wire config/symbols/cleaners + monotonic_align stub
    import setup_vits  # same folder
    return setup_vits.main()


if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    sys.exit(main())
