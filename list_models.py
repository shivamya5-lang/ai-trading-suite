import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from google import genai

key = os.getenv("GEMINI_API_KEY", "").strip()
if not key:
    raise SystemExit("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=key)

print(f"{'MODEL NAME':<45} SUPPORTED METHODS")
print("-" * 80)

for m in client.models.list():
    name = getattr(m, "name", str(m))
    # attribute name differs across SDK versions — try both
    methods = (
        getattr(m, "supported_generation_methods", None)
        or getattr(m, "supported_actions", None)
        or []
    )
    marker = " *** generateContent" if "generateContent" in methods else ""
    print(f"{name:<45} {methods}{marker}")
