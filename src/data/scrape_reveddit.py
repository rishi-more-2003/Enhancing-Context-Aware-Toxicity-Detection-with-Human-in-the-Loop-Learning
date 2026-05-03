"""Scrape moderator-removed comments from Reveddit using Playwright.

These comments form the *golden-standard* evaluation set: content that
human moderators deemed toxic enough to remove.

Usage::

    python -m src.data.scrape_reveddit \
        --output data/golden_standard/removed_subreddit_comments.csv
"""

import argparse
import asyncio

import pandas as pd
from playwright.async_api import async_playwright


async def scrape_reveddit(
    output_path: str = "data/golden_standard/removed_subreddit_comments.csv",
    n_comments: int = 1000,
) -> pd.DataFrame:
    """Launch a headless browser, scrape Reveddit, and save to CSV."""

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True, args=["--disable-dev-shm-usage"]
        )
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        page = await context.new_page()

        try:
            url = (
                f"https://www.reveddit.com/v/politics/history/"
                f"?showFilters=true&n={n_comments}"
            )
            print("Loading Reveddit page...")
            await page.goto(url, wait_until="networkidle")

            # Dismiss popup if present
            try:
                modal = await page.wait_for_selector("#genericModal", timeout=5000)
                if modal:
                    btn = await page.query_selector("#genericModal .dismiss a")
                    if btn:
                        await btn.click()
                        print("Popup dismissed")
            except Exception:
                pass

            print("Waiting for comments...")
            await page.wait_for_selector(".comment-body", timeout=60000)
            await asyncio.sleep(5)

            comments = await page.query_selector_all(".comment-body")
            comment_data: list[dict] = []

            print(f"Found {len(comments)} comments. Extracting...")
            for comment in comments:
                try:
                    text = await comment.evaluate(
                        """el => {
                            const p = el.querySelector('div > p');
                            return p ? p.innerText : '';
                        }"""
                    )
                    comment_data.append({"text": text, "toxic": 1})
                except Exception as exc:
                    print(f"  Extraction error: {exc}")

            if comment_data:
                df = pd.DataFrame(comment_data)
                df.to_csv(output_path, index=False, encoding="utf-8")
                print(f"Saved {len(df)} comments to {output_path}")
                return df

            print("No comments found")
            return pd.DataFrame()

        finally:
            await browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Reveddit mod-removed comments")
    parser.add_argument(
        "--output",
        default="data/golden_standard/removed_subreddit_comments.csv",
    )
    parser.add_argument("--n", type=int, default=1000)
    args = parser.parse_args()

    asyncio.run(scrape_reveddit(output_path=args.output, n_comments=args.n))


if __name__ == "__main__":
    main()
