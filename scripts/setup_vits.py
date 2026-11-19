"""One-time wiring of the VITS inference environment (project-local, no caches).

Run:  python scripts\setup_vits.py
- copies nepali symbols/cleaners/config into the cloned jaywalnut310/vits tree
- creates a monotonic_align stub (the C extension is training-only)
- verifies files are valid UTF-8 Devanagari
"""

import os
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
VITS = ROOT / "third_party" / "vits"
MODELS = ROOT / "models" / "vits"


def main() -> int:
    if not VITS.is_dir():
        print("third_party/vits missing - run the model download first")
        return 1

    # 1. copy config/symbols/cleaners into the vits tree (binary-safe copy)
    shutil.copy(MODELS / "nepali_symbols.py", VITS / "text" / "symbols.py")
    shutil.copy(MODELS / "nepali_cleaners.py", VITS / "text" / "cleaners.py")
    (VITS / "configs").mkdir(exist_ok=True)
    shutil.copy(MODELS / "nepali_base.json", VITS / "configs" / "nepali_base.json")
    print("copied: text/symbols.py, text/cleaners.py, configs/nepali_base.json")

    # 2. verify UTF-8 Devanagari survived (PowerShell would have mangled it)
    src = (VITS / "text" / "symbols.py").read_text(encoding="utf-8")
    if "क" not in src:
        print("ERROR: symbols.py is not valid UTF-8 Devanagari")
        return 1
    print("verified: symbols.py contains Devanagari")

    # 3. monotonic_align stub (models.py imports it at module level, but the
    #    call is training-only; inference never needs the compiled extension)
    stub_dir = VITS / "monotonic_align"
    if not (stub_dir / "core.pyd").exists():
        stub_dir.mkdir(exist_ok=True)
        (stub_dir / "__init__.py").write_text(
            '"""Pure-Python stub for monotonic_align (jaywalnut310/vits).\n'
            "\n"
            "The compiled extension is used only during TRAINING (alignment\n"
            "via maximum_path). net_g.infer() never calls it, so an importable\n"
            "stub removes the need for Cython + MSVC on Windows.\n"
            '"""\n'
            "\n"
            "def maximum_path(*args, **kwargs):\n"
            '    raise RuntimeError("monotonic_align is training-only; inference '
            'does not need alignments.")\n',
            encoding="utf-8")
        print("created: monotonic_align/__init__.py stub")
    else:
        print("compiled monotonic_align present - stub skipped")

    # 4. keep every cache inside the project (belt & braces for torch/HF)
    cache = ROOT / "models" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    print(f"cache env (for child processes): TORCH_HOME/HF_HOME -> {cache}")
    print("setup OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
