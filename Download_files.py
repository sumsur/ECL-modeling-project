from pathlib import Path
from urllib.request import urlopen, Request
from zipfile import ZipFile, ZIP_DEFLATED
import csv

# Public community-prepared subset of Freddie Mac loan-level data.
# Source repository states that it contains 9,572 loans originated in Q1 2020.
FILES = {
    "origination.csv": (
        "https://raw.githubusercontent.com/akmalf/freddie-mac-sample/main/sampler.csv"
    ),
    "performance.csv": (
        "https://raw.githubusercontent.com/akmalf/freddie-mac-sample/main/performance_sampler.csv"
    ),
}

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "data_raw" / "freddie_mac_sample"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def download_file(url: str, destination: Path) -> None:
    print(f"Downloading {destination.name} ...")
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=120) as response, destination.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)

    if destination.stat().st_size == 0:
        raise RuntimeError(f"Downloaded file is empty: {destination.name}")

    print(f"Saved: {destination}")
    print(f"Size: {destination.stat().st_size / (1024 * 1024):.2f} MB")

for filename, url in FILES.items():
    download_file(url, OUTPUT_DIR / filename)

# Basic validation of the origination file.
origination_path = OUTPUT_DIR / "origination.csv"
with origination_path.open("r", encoding="utf-8-sig", newline="") as file:
    reader = csv.reader(file)
    header = next(reader)
    row_count = sum(1 for _ in reader)

print("\nORIGINATION FILE CHECK")
print("Columns:", len(header))
print("Loan records:", row_count)
print("First columns:", header[:10])

# Package both raw files and source notes into one ZIP.
readme = OUTPUT_DIR / "SOURCE_NOTES.txt"
readme.write_text(
    "FREDDIE MAC LOAN-LEVEL SAMPLE\n"
    "============================\n\n"
    "Files:\n"
    "- origination.csv: loan origination characteristics\n"
    "- performance.csv: monthly loan performance records\n\n"
    "Source repository:\n"
    "https://github.com/akmalf/freddie-mac-sample\n\n"
    "The repository README states that this is a subset of Freddie Mac's\n"
    "Single-Family Loan-Level Dataset containing 9,572 loans originated\n"
    "in Q1 2020, with performance data available through 2022 Q2.\n\n"
    "This is a third-party prepared subset, not a direct official Freddie\n"
    "Mac download. Retain this source note and review applicable terms\n"
    "before redistribution or commercial use.\n",
    encoding="utf-8"
)

zip_path = OUTPUT_DIR / "freddie_mac_sample.zip"
with ZipFile(zip_path, "w", ZIP_DEFLATED) as archive:
    archive.write(OUTPUT_DIR / "origination.csv", arcname="origination.csv")
    archive.write(OUTPUT_DIR / "performance.csv", arcname="performance.csv")
    archive.write(readme, arcname="SOURCE_NOTES.txt")

print("\nDOWNLOAD COMPLETED")
print("Folder:", OUTPUT_DIR)
print("ZIP:", zip_path)
print("Next: use origination.csv and performance.csv as separate source tables.")
