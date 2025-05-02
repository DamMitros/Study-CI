import os
import praw
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def get_reddit_data(topic, limit=100):
  reddit = praw.Reddit(
    client_id=os.getenv('REDDIT_CLIENT_ID'),
    client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
    user_agent=os.getenv('REDDIT_USER_AGENT'),
    read_only=True
  )

  print("Collecting data from Reddit...")
  posts = []
  for submission in reddit.subreddit('all').search(topic, limit=limit, sort='relevance'):
    posts.append({
      'id': submission.id,
      'title': submission.title,
      'author': str(submission.author),
      'subreddit': submission.subreddit.display_name,
      'text': submission.selftext,
      'score': submission.score,
      'created_utc': pd.to_datetime(submission.created_utc, unit='s'),
      'url': submission.url,
    })

  print(f"Collected {len(posts)} posts from Reddit.")
  return pd.DataFrame(posts)

def save_to_csv(dataframe, filename):
  dataframe.to_csv(filename, index=False, encoding='utf-8')
  print(f"Data saved to {filename}.")

def main():
  topic = "Spain Power Outage"
  data = get_reddit_data(topic)
  save_to_csv(data, f"{topic.replace(' ', '_')}.csv")

main()