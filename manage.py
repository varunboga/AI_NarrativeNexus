import praw
import pandas as pd
from datetime import datetime, timezone

def fetch_multiple_subreddits(subreddits, limit=50, filename="reddit_posts.csv"):
    reddit = praw.Reddit(
        client_id="HgDHCgxpwDNQyPbuyzxoSg",
        client_secret="UwzrM8Obvjk0_PO4HbcDNfC4Cs617g",
        user_agent="NarrativeNexus by u/varunboga"
    )

    posts = []

    for sub in subreddits:
        for post in reddit.subreddit(sub).hot(limit=limit):
            record = {
                "id": post.id,
                "source": "reddit",
                "author": post.author.name if post.author else "unknown",
                "timestamp": datetime.fromtimestamp(post.created_utc, tz=timezone.utc).isoformat(),
                "text": f"{post.title}\n{post.selftext}",
                "metadata": {
                    "subreddit": sub,
                    "language": "en",
                    "likes": post.score,
                    "rating": None,
                    "url": f"https://www.reddit.com{post.permalink}"
                }
            }
            posts.append(record)

    df = pd.json_normalize(posts)
    df.to_csv(filename, index=False, encoding="utf-8")
    print(f"Saved {len(posts)} posts from {len(subreddits)} subreddits to {filename}")
    return df

fetch_multiple_subreddits(["MachineLearning", "datascience", "artificial"], limit=50)
