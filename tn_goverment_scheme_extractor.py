"""
Tamil Nadu Government Schemes Web Scraper
=========================================
Extracts schemes data from the Government of Tamil Nadu portal using Playwright.

Workflow:
  1. Navigate to base URL: https://www.tn.gov.in/schemes.php
  2. Extract all departments from class="dept_container" (name and href link).
  3. For each department, visit its hyperlink and extract all schemes inside
     class="custom-innersection" with id="content" (header/title and scheme href).
  4. For each scheme, visit its hyperlink and extract scheme details from
     table class="table table-condensed table-hover table-striped ng-table".
  5. Save the extracted data into a structured JSON file.
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright

BASE_URL = "https://www.tn.gov.in/schemes.php"
DEFAULT_OUTPUT_FILE = "tn_government_schemes.json"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("TN-Scheme-Extractor")


def save_json(data: Any, filepath: str) -> None:
    """Save data to JSON file with indentation and UTF-8 encoding."""
    temp_path = f"{filepath}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(temp_path, filepath)
    logger.info(f"Saved extracted data to: {filepath}")


def navigate_and_wait_until_loaded(
    page: Page,
    url: str,
    timeout: int = 60000,
    wait_selector: Optional[str] = None,
) -> None:
    """
    Navigates to the given URL and ensures the page is completely loaded:
      1. Waits for window 'load' event (HTML, styles, scripts, assets).
      2. Waits for 'networkidle' state (no network activity for at least 500ms).
      3. Optionally waits for a specific content selector to become visible.
    """
    page.goto(url, wait_until="load", timeout=timeout)
    try:
        page.wait_for_load_state("networkidle", timeout=15000)
    except Exception:
        # Fallback if long-lived analytics or polling connection is active
        pass

    if wait_selector:
        try:
            page.wait_for_selector(wait_selector, state="visible", timeout=timeout)
        except Exception as e:
            logger.warning(f"Selector '{wait_selector}' not visible yet: {e}")


def fetch_departments(page: Page, base_url: str = BASE_URL) -> List[Dict[str, str]]:
    """
    Step 1 & 2:
    Navigate to schemes.php, wait until completely loaded, and extract
    all departments from class="dept_container".
    """
    logger.info(f"Navigating to base URL: {base_url}")
    navigate_and_wait_until_loaded(page, base_url, wait_selector=".dept_container")

    departments: List[Dict[str, str]] = []
    dept_elements = page.locator(".dept_container a").all()

    for el in dept_elements:
        name = el.inner_text().strip()
        href = el.get_attribute("href")
        if href and name:
            full_url = urljoin(page.url, href)
            departments.append({"name": name, "href": full_url})

    logger.info(f"Found {len(departments)} departments under '.dept_container'.")
    return departments


def fetch_schemes_for_department(
    page: Page, department_url: str
) -> List[Dict[str, str]]:
    """
    Step 3:
    Navigate to department URL, wait until page is completely loaded,
    and extract all schemes inside .custom-innersection #content.
    """
    navigate_and_wait_until_loaded(
        page,
        department_url,
        wait_selector=".custom-innersection #content, #content",
    )

    content_container = page.locator(".custom-innersection #content, #content").first
    if not content_container.is_visible():
        logger.warning(f"No #content container found at {department_url}")
        return []

    scheme_links = content_container.locator("a").all()
    schemes: List[Dict[str, str]] = []

    for link in scheme_links:
        title = link.inner_text().strip()
        href = link.get_attribute("href")
        if href and title:
            full_url = urljoin(page.url, href)
            schemes.append({"title": title, "href": full_url})

    return schemes


def fetch_scheme_details(page: Page, scheme_url: str) -> Dict[str, Any]:
    """
    Step 4:
    Navigate to scheme details URL, wait until completely loaded,
    and extract table details from class="table table-condensed table-hover table-striped ng-table".
    """
    table_selector = "table.table-condensed.table-hover.table-striped.ng-table, table.table"
    navigate_and_wait_until_loaded(page, scheme_url, wait_selector=table_selector)

    table = page.locator(table_selector).first
    details: Dict[str, Any] = {}

    if not table.is_visible():
        logger.warning(f"No details table found at: {scheme_url}")
        return details

    rows = table.locator("tr").all()
    for row in rows:
        cols = row.locator("td, th").all()
        if len(cols) == 2:
            key = cols[0].inner_text().strip().rstrip(":")
            val_text = cols[1].inner_text().strip()

            # Check if there is an embedded hyperlink in the value column (e.g. document/form attachment)
            link_el = cols[1].locator("a").first
            if link_el.count() > 0:
                href = link_el.get_attribute("href")
                if href and href.strip() and href != "#":
                    full_file_url = urljoin(page.url, href.strip())
                    details[key] = {
                        "text": val_text if val_text else "Download Link",
                        "url": full_file_url,
                    }
                    continue

            if key:
                details[key] = val_text

        elif len(cols) == 1:
            # Single row / section title or header
            header_text = cols[0].inner_text().strip()
            if header_text and not details.get("Department Header"):
                details["Department Header"] = header_text

    return details


def scrape_all_schemes(
    headless: bool = True,
    output_file: str = DEFAULT_OUTPUT_FILE,
    delay: float = 0.3,
    max_departments: Optional[int] = None,
    max_schemes_per_dept: Optional[int] = None,
    block_media: bool = False,
) -> List[Dict[str, Any]]:
    """
    Main orchestration function:
    Iterates through all departments, schemes, and details, and stores them in JSON.
    Waits for full page load at each step before performing any extraction.
    """
    results: List[Dict[str, Any]] = []
    total_schemes_count = 0

    with sync_playwright() as p:
        browser: Browser = p.chromium.launch(headless=headless)
        context: BrowserContext = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        )

        page: Page = context.new_page()

        # Optional media blocking (disabled by default to ensure complete page load)
        if block_media:
            page.route(
                "**/*.{png,jpg,jpeg,gif,webp,svg,woff,woff2,ttf,otf,css}",
                lambda route: route.abort(),
            )

        try:
            # Step 1 & 2: Fetch departments
            departments = fetch_departments(page, base_url=BASE_URL)
            if max_departments:
                departments = departments[:max_departments]
                logger.info(f"Limiting to first {max_departments} departments for this run.")

            # Iterate through each department
            for dept_idx, dept in enumerate(departments, 1):
                dept_name = dept["name"]
                dept_href = dept["href"]
                logger.info(
                    f"\n[{dept_idx}/{len(departments)}] Processing Department: {dept_name}"
                )

                dept_entry: Dict[str, Any] = {
                    "department_name": dept_name,
                    "department_url": dept_href,
                    "schemes": [],
                }

                # Step 3: Fetch schemes for this department
                try:
                    schemes = fetch_schemes_for_department(page, dept_href)
                    logger.info(f"  Found {len(schemes)} scheme(s) in {dept_name}")
                except Exception as e:
                    logger.error(f"  Error fetching schemes for {dept_name}: {e}")
                    schemes = []

                if max_schemes_per_dept:
                    schemes = schemes[:max_schemes_per_dept]

                # Step 4: Fetch details for each scheme
                for scheme_idx, scheme in enumerate(schemes, 1):
                    scheme_title = scheme["title"]
                    scheme_href = scheme["href"]
                    logger.info(
                        f"  -> [{scheme_idx}/{len(schemes)}] Scraping scheme: {scheme_title}"
                    )

                    try:
                        details = fetch_scheme_details(page, scheme_href)
                    except Exception as e:
                        logger.error(f"     Failed scraping {scheme_title} ({scheme_href}): {e}")
                        details = {"error": str(e)}

                    dept_entry["schemes"].append(
                        {
                            "scheme_title": scheme_title,
                            "scheme_url": scheme_href,
                            "details": details,
                        }
                    )
                    total_schemes_count += 1

                    if delay > 0:
                        time.sleep(delay)

                results.append(dept_entry)

                # Periodic save after each department so progress is never lost
                save_json(results, output_file)

        finally:
            browser.close()

    logger.info("\n" + "=" * 60)
    logger.info(
        f"Scraping completed! Total departments: {len(results)}, "
        f"Total schemes extracted: {total_schemes_count}"
    )
    logger.info(f"Final data saved to: {output_file}")
    logger.info("=" * 60)
    return results


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Tamil Nadu Government Schemes using Playwright."
    )
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output JSON file path (default: {DEFAULT_OUTPUT_FILE})",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Run browser in headful (visible) mode for debugging.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="Politeness delay in seconds between scheme requests (default: 0.3s).",
    )
    parser.add_argument(
        "--max-departments",
        type=int,
        default=None,
        help="Limit number of departments to scrape (useful for quick testing).",
    )
    parser.add_argument(
        "--max-schemes-per-dept",
        type=int,
        default=None,
        help="Limit number of schemes per department to scrape (useful for testing).",
    )
    parser.add_argument(
        "--block-media",
        action="store_true",
        help="Block images and fonts to save bandwidth.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    scrape_all_schemes(
        headless=False,
        output_file=args.output,
        delay=args.delay,
        max_departments=args.max_departments,
        max_schemes_per_dept=args.max_schemes_per_dept,
        block_media=args.block_media,
    )


if __name__ == "__main__":
    main()
