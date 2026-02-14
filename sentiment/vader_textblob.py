"""
VADER and TextBlob Sentiment Analysis Demo

Demonstrates sentence-level sentiment scoring using two popular
lexicon-based approaches.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob


def main():
    analyzer = SentimentIntensityAnalyzer()

    sentences = [
        "This is a good course",
        "This is an awesome course",
        "The instructor is so cool",
        "The instructor is so cool!!",
        "The instructor is so COOL!!",
        "Machine learning makes me :)",
        "His antics had me ROFL",
        "The movie SUX",
    ]

    print("=== VADER Sentiment Scores ===")
    for s in sentences:
        score = analyzer.polarity_scores(s)
        print(f"  {s}")
        print(f"  → {score}\n")

    # TextBlob demo
    words = ["His", "remarkable", "work", "ethic", "impressed", "me"]
    sentence = "His remarkable work ethic impressed me"

    print("=== TextBlob Sentiment Scores ===")
    for word in words:
        sentiment = TextBlob(word).sentiment
        print(f"  {word}: {sentiment}")

    print(f"\n  Sentence: '{sentence}'")
    print(f"  Sentiment: {TextBlob(sentence).sentiment}")


if __name__ == '__main__':
    main()
