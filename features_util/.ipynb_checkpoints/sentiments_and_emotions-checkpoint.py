from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#import utils

# Vader for sentiment analysis
def get_vader_scores(text):
    analyzer = SentimentIntensityAnalyzer()
    def translate_scores(text):
        score = analyzer.polarity_scores(text)
        compound = score['compound']

        sentiment = 'neutral'
        if compound >= 0.05:
            sentiment = "positive"
        elif compound <= -0.05:
            sentiment = "negative"
        return sentiment    
    return translate_scores(text)

#emotions = utils.Models.emotion_model()

# Compute the emotions inside of a given text
def emotion_classification(text):
    return emotions(text)[0]['label']