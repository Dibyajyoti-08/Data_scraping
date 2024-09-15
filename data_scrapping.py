import tweepy

consumer_key =  ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

# Authenticate with
auth = tweepy.0Auth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
api tweepy.API(auth)


# Scrape tweets
def fetch_tweets(keyword, count=100):
    tweets = tweepy.Cursor(api.search_tweets, q=keyword, lang="en", tweet_mode='extended').items(count)
    tweet_list = [[tweet.full_text, tweet.created_at] for tweet in tweets]
    return tweet_list

tweets = fetch_tweets("Python", 200)
for tweet in tweets[:5]:
    print(tweet)
