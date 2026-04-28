"""
scripts/setup_check.py
──────────────────────
Run this to verify your environment is ready before starting.
    python scripts/setup_check.py
"""

import sys
import os

OK   = "(OK)"
FAIL = "(FAIL)"
WARN = "(WARN)"

print("\nAI Travel Assistant Environment Check\n" + "─" * 45)

# 1. Python version
major, minor = sys.version_info[:2]
if major == 3 and minor >= 9:
    print(f"{OK} Python {major}.{minor} (3.9+ required)")
else:
    print(f"{FAIL} Python {major}.{minor} — need 3.9 or higher")

# 2. .env file
from pathlib import Path
env_path = Path(".env")
if env_path.exists():
    print(f"{OK} .env file found")
else:
    print(f"{FAIL} .env not found — run: cp .env.example .env  then fill in your keys")

# 3. Required packages
packages = {
    "langchain":             "langchain",
    "langchain_community":   "langchain-community",
    "langchain_huggingface": "langchain-huggingface",
    "pinecone":              "pinecone-client",
    "sentence_transformers": "sentence-transformers",
    "pypdf":                 "pypdf",
    "streamlit":             "streamlit",
    "dotenv":                "python-dotenv",
}

print("\nPackages:")
all_ok = True
for module, pkg in packages.items():
    try:
        __import__(module)
        print(f"  {OK} {pkg}")
    except ImportError:
        print(f"  {FAIL} {pkg}  →  pip install {pkg}")
        all_ok = False

# 4. Environment variables
print("\nEnvironment variables:")
from dotenv import load_dotenv
load_dotenv()

keys = {
    "PINECONE_API_KEY":       "required",
    "PINECONE_INDEX_NAME":    "required",
    "HUGGINGFACE_API_TOKEN":  "required",
    "OPENAI_API_KEY":         "optional",
}

for key, status in keys.items():
    val = os.getenv(key)
    if val:
        masked = val[:6] + "…" + val[-3:] if len(val) > 9 else "***"
        print(f"  {OK} {key} = {masked}")
    elif status == "optional":
        print(f"  {WARN} {key} not set (optional — only needed for GPT)")
    else:
        print(f"  {FAIL} {key} not set — add it to your .env file")

# 5. PDF
print("\nTravel guide PDF:")
pdf = os.getenv("PDF_PATH", "data/paris_guide.pdf")
if Path(pdf).exists():
    size = Path(pdf).stat().st_size // 1024
    print(f"  {OK} {pdf} ({size} KB)")
else:
    print(f"  {FAIL} {pdf} not found — place your PDF in the data/ folder")

# 6. Summary
print("\n" + "─" * 45)
if all_ok:
    print("All packages installed! Next steps:")
    print("   1. Fill in .env with your API keys")
    print("   2. Add paris_guide.pdf to data/")
    print("   3. Run: python backend/ingest.py")
    print("   4. Run: streamlit run frontend/app.py")
else:
    print("Some packages are missing. Run:")
    print("   pip install -r requirements.txt")
print()
