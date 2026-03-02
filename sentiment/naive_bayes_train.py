"""
Naive Bayes Classifier for Crude Oil Sentiment

Trains a Multinomial Naive Bayes on crude oil news articles, evaluates
performance, and saves the model + vectorizer to data/.
"""

import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
)


def convert_score_to_label(score):
    """Convert continuous sentiment score to categorical label."""
    if score > 0.05:
        return "positive"
    elif score < -0.05:
        return "negative"
    return "neutral"


def train_naive_bayes(csv_file, output_dir='data'):
    """
    Train and evaluate a Naive Bayes classifier on text data.

    Args:
        csv_file: Path to CSV with columns: [*, *, News, Sentiment].
        output_dir: Directory to save model artifacts.
    """
    data = pd.read_csv(csv_file, encoding='utf-8')
    X = data.iloc[:, 2]  # News text
    y = data.iloc[:, 3].apply(convert_score_to_label)

    print("Class distribution:")
    print(y.value_counts(), "\n")

    vectorizer = CountVectorizer(stop_words='english')
    X_vec = vectorizer.fit_transform(X)

    tfidf = TfidfTransformer()
    X_tfidf = tfidf.fit_transform(X_vec)

    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42
    )

    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("=== Evaluation Results ===")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")

    cm = confusion_matrix(y_test, y_pred, labels=["negative", "neutral", "positive"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=["negative", "neutral", "positive"])
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

    os.makedirs(output_dir, exist_ok=True)
    pickle.dump(vectorizer, open(os.path.join(output_dir, 'vectorizer_crude_oil.pkl'), 'wb'))
    pickle.dump(clf, open(os.path.join(output_dir, 'naive_bayes_classifier_crude_oil.pkl'), 'wb'))
    print(f"Model saved to {output_dir}/")


if __name__ == '__main__':
    train_naive_bayes('data/crude_oil_news_articles.csv')
