"""
NLP Basics Demo

Demonstrates tokenization, lemmatization, stemming, and stop word removal
using NLTK.
"""

import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords


def main():
    # Download required resources
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('stopwords', quiet=True)

    text = 'I am not a sentimential person but I believe in the utility of sentiment analysis'

    tokenizer = TreebankWordTokenizer()
    tokens = tokenizer.tokenize(text)

    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]

    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(word.lower()) for word in tokens]

    stop_words = set(stopwords.words('english'))
    filtered = [word for word in stemmed if word not in stop_words]

    print('Tokens:    ', tokens)
    print('Lemmatized:', lemmatized)
    print('Stemmed:   ', stemmed)
    print('Filtered:  ', filtered)


if __name__ == '__main__':
    main()
