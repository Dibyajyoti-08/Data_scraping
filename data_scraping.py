import tweepy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

consumer_key =  ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

# Authenticate with Twitter APi
auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
api = tweepy.API(auth)


# Scrape tweets
def fetch_tweets(keyword, count=100):
    try:
        tweets = tweepy.Cursor(api.search_tweets, q=keyword, lang="en", tweet_mode='extended').items(count)
        tweet_list = [[tweet.full_text, tweet.created_at] for tweet in tweets]
        return tweet_list
    except tweepy.TweepError as e:
        print(f"Error fetching tweets: {e}")

tweets = fetch_tweets("Python", 200)
for tweet in tweets[:5]:
    print(tweet)


# Process tweet data
def clean_tweet(tweet):
    # Remove URLs, mentions, and hashtags
    tweet = re.sub(r'http\S+|@\S+|#\S+', '', tweet)

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

# Sentiment Analysis Model Build
labels = [0 if 'bad' in tweet else 1 for tweet in cleaned_tweets]

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))