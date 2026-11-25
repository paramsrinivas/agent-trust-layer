"""Quick environment check for Agent Trust Layer dependencies and artifacts."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

REQUIRED = ["numpy", "pandas", "sklearn", "streamlit"]
OPTIONAL = ["razorpay", "dotenv"]

missing = []
for pkg in REQUIRED:
    try:
        mod = importlib.import_module(pkg)
        print(f"OK   {pkg:12s} {getattr(mod, '__version__', 'installed')}")
    except ImportError:
        print(f"MISS {pkg:12s} (required)")
        missing.append(pkg)

for pkg in OPTIONAL:
    try:
        mod = importlib.import_module(pkg)
        print(f"OK   {pkg:12s} {getattr(mod, '__version__', 'installed')} (optional for offline mode)")
    except ImportError:
        print(f"WARN {pkg:12s} not installed (dry-run mode only)")

artifacts = [
    Path("data/population_baseline.json"),
    Path("data/experiments/trust_model_A.pkl"),
    Path("data/experiments/trust_model_B.pkl"),
]
for art in artifacts:
    status = "OK  " if art.exists() else "WARN"
    print(f"{status} artifact     {art}")

if missing:
    print(f"\nInstall missing packages: pip install -r requirements.txt")
    sys.exit(1)

print("\nEnvironment ready.")
