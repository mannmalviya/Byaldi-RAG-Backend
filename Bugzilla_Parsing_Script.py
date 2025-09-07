#!/usr/bin/env python3
import requests
from bs4 import BeautifulSoup
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE_URL = "http://bugzilla.asicdesigners.com/bugs/show_bug.cgi?id="

# Directory to store JSON files
OUTPUT_DIR = Path("../ALL_DATA/Bugzilla")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def fetch_and_parse_bug(bug_id: int) -> dict | None:
    """Fetch a Bugzilla page and parse its details into a dict."""
    url = f"{BASE_URL}{bug_id}"
    try:
        resp = requests.get(url, timeout=100)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[ERROR] Bug {bug_id} request failed: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # --- Extract summary ---
    summary = None
    summary_el = soup.find("span", id="short_desc_nonedit_display")
    if summary_el:
        summary = summary_el.get_text(strip=True)
    elif soup.title:
        summary = soup.title.get_text(strip=True)

    # --- Extract status ---
    status = None

    # Preferred: explicit id lookup
    status_el = soup.find("span", id="static_bug_status")
    if status_el:
        status = status_el.get_text(" ", strip=True)


    # --- Extract Product ---
    product = None
    product_el = soup.find("td", id="field_container_product")
    if product_el:
        product = product_el.get_text(" ", strip=True)

    # --- Extract Version ---
    version = None
    ver_th = soup.find("th", id="field_label_version")
    if ver_th:
        td = ver_th.find_next("td")
        if td:
            version = td.get_text(" ", strip=True)

    # --- Extract description ---
    description = None
    desc_el = soup.find("pre", class_="bz_comment_text")
    if desc_el:
        description = desc_el.get_text("\n", strip=True)


    # --- Extract comments ---
    comments = []
    for cdiv in soup.find_all("div", class_="bz_comment"):
        text = cdiv.find("pre", class_="bz_comment_text")
        if text:
            comments.append(text.get_text("\n", strip=True))

    bug_data = {
        "id": bug_id,
        "product":product,
        "version":version,
        "summary": summary,
        "status": status,
        "description": description,
        "comments": comments,
    }

    print(f"Bug {bug_id}: product={product}, version={version}")

    # Save to file
    out_file = OUTPUT_DIR / f"bug_{bug_id}.json"
    with open(out_file, "w") as f:
        json.dump(bug_data, f, indent=2)

    print(f"[SAVED] bug_{bug_id}.json")
    return bug_data


def fetch_range(start: int, end: int, max_workers: int = 8):
    """Fetch and parse a range of bugs in parallel using multiple cores."""
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_and_parse_bug, bug_id): bug_id for bug_id in range(start, end + 1)}
        for future in as_completed(futures):
            bug_id = futures[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"[ERROR] Bug {bug_id} processing failed: {e}")
    return results


if __name__ == "__main__":
    # Example: scrape bugs 60–1000
    fetch_range(21650, 45000, max_workers=40)
