"""Scrape Reddit comments from r/politics using asyncpraw.

Requires a valid Reddit API application. Set credentials via environment
variables or pass them directly.

Usage::

    python -m src.data.scrape_reddit \
        --client-id YOUR_ID \
        --client-secret YOUR_SECRET \
        --output data/subreddit_comments.csv
"""

import argparse
import asyncio
from datetime import datetime, timezone

import asyncpraw
import asyncpraw.models
import pandas as pd

from config import SUBREDDIT, SEARCH_QUERIES, COMMENTS_PER_QUERY


async def scrape_comments(
    client_id: str,
    client_secret: str,
    user_agent: str = "toxicity_scraper",
    subreddit_name: str = SUBREDDIT,
    search_queries: list[str] | None = None,
    comments_per_query: int = COMMENTS_PER_QUERY,
    output_path: str = "data/subreddit_comments.csv",
) -> pd.DataFrame:
    """Collect comments from Reddit and save to CSV."""

    if search_queries is None:
        search_queries = SEARCH_QUERIES

    reddit = asyncpraw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )

    subreddit = await reddit.subreddit(subreddit_name)
    comments_data: list[dict] = []

    for query in search_queries:
        collected = 0
        print(f"\nStarting collection for query: {query}")

        async for submission in subreddit.search(query=query, sort="hot", limit=None):
            try:
                full_submission = await reddit.submission(id=submission.id)
                if not full_submission.comments:
                    continue

                await full_submission.comments.replace_more(limit=None)
                for comment in full_submission.comments.list():
                    if not isinstance(comment, asyncpraw.models.Comment):
                        continue

                    ups = comment.ups
                    downs = abs(comment.downs)
                    total = ups + downs if (ups + downs) > 0 else 1

                    comments_data.append(
                        {
                            "comment_id": comment.id,
                            "submission_id": submission.id,
                            "author": str(comment.author),
                            "created_utc": datetime.fromtimestamp(
                                comment.created_utc, tz=timezone.utc
                            ).isoformat(),
                            "body": comment.body,
                            "upvote_ratio": ups / total,
                            "downvote_ratio": downs / total,
                            "score": comment.score,
                            "ups": ups,
                            "downs": downs,
                            "search_query": query,
                        }
                    )
                    collected += 1

                    if collected % 500 == 0:
                        print(f"  Collected {collected} comments for '{query}'")
                    if collected >= comments_per_query:
                        break

            except Exception as exc:
                print(f"  Error on submission {submission.id}: {exc}")
                continue

            if collected >= comments_per_query:
                print(f"  Completed '{query}' with {collected} comments")
                break

    df = pd.DataFrame(comments_data)
    df.to_csv(output_path, index=False)
    print(f"\nTotal comments scraped: {len(df)}")
    print(df["search_query"].value_counts())
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Reddit comments")
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--client-secret", required=True)
    parser.add_argument("--output", default="data/subreddit_comments.csv")
    args = parser.parse_args()

    asyncio.run(
        scrape_comments(
            client_id=args.client_id,
            client_secret=args.client_secret,
            output_path=args.output,
        )
    )


if __name__ == "__main__":
    main()
