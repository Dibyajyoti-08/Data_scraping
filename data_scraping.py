import tweepy
import re
from sklearn.feature_extraction.text import TfidfVectorizer

consumer_key =  ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

# Authenticate with
auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
api = tweepy.API(auth)


# Scrape tweets
def fetch_tweets(keyword, count=100):
    tweets = tweepy.Cursor(api.search_tweets, q=keyword, lang="en", tweet_mode='extended').items(count)
    tweet_list = [[tweet.full_text, tweet.created_at] for tweet in tweets]
    return tweet_list

tweets = fetch_tweets("Python", 200)
for tweet in tweets[:5]:
    print(tweet)


# Process tweet data
def clean_tweet(tweet):
    # Remove URLs, mentions and hashtag
    tweet = re.sub(r'http\S+|@\S+|#\S', '', tweet)

    # Remove special characters and numbers
    tweet = re.sub(r'[^A-Za-z\s]', '', tweet)

    # Convert to lowercase
    tweet = tweet.lower()
    
    return tweet

# Apply the processing
cleaned_tweets = [clean_tweet(tweet[0]) for tweet in tweets]

# Tokenization and Vectorization
'''
Convertion of cleaned tweets into numerical 
representations using TF-IDF or Tokenizer.
'''
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(cleaned_tweets).toarray()