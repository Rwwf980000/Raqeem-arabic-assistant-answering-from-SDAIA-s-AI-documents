"""Download the SDAIA AI documents into data/.

Run this on your own machine. Any PDF you put in data/ by hand is indexed too,
so add the remaining documents from
https://sdaia.gov.sa/en/SDAIA/about/Pages/AboutAI.aspx the same way.
"""
import urllib.request

from config import DATA_DIR

DOCS = {
    "ai_ethics_principles.pdf": "https://sdaia.gov.sa/en/SDAIA/about/Documents/ai-principles.pdf",
    "genai_guidelines_public.pdf": "https://sdaia.gov.sa/en/SDAIA/about/Files/GenerativeAIPublicEN.pdf",
    "genai_guidelines_government.pdf": "https://sdaia.gov.sa/en/SDAIA/about/Files/GenAIGuidelinesForGovernmentENCompressed.pdf",
    "ai_adoption_framework.pdf": "https://sdaia.gov.sa/en/SDAIA/about/Files/AIAdoptionFramework.pdf",
    "national_ai_index.pdf": "https://sdaia.gov.sa/en/SDAIA/about/Files/NAII.pdf",
    "deepfakes_guidelines.pdf": "https://sdaia.gov.sa/en/SDAIA/about/Files/File0001.pdf",
}

HEADERS = {"User-Agent": "Mozilla/5.0"}  # the site rejects requests without one


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in DOCS.items():
        path = DATA_DIR / name
        if path.exists():
            print(f"skip   {name} (already downloaded)")
            continue
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read()
            if not body.startswith(b"%PDF"):
                raise ValueError("response is not a PDF (blocked or moved?)")
            path.write_bytes(body)
            print(f"saved  {name} ({len(body) // 1024} KB)")
        except Exception as e:
            print(f"FAILED {name}: {e}\n       download it in a browser from {url} and save it as data/{name}")


if __name__ == "__main__":
    main()
