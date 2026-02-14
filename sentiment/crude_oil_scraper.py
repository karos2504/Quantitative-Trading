"""
Crude Oil News Scraper with Sentiment Analysis

Scrapes crude oil news articles from oilprice.com and scores them
using VADER sentiment analysis.
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def get_article_urls(base_url, pages):
    """Scrape article URLs from listing pages."""
    urls = []
    for page_num in range(1, pages + 1):
        page_url = f"{base_url}/Energy/Crude-Oil/Page-{page_num}.html"
        response = requests.get(page_url)
        soup = BeautifulSoup(response.text, 'html.parser')

        for div in soup.find_all('div', class_='categoryArticle'):
            for link in div.find_all('a', href=True):
                url = link['href']
                if url not in urls:
                    urls.append(url)
    return urls


def scrape_article_data(url):
    """Scrape headline, date, and content from an article page."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    headline_tag = soup.find('h1')
    headline = headline_tag.text.strip() if headline_tag else url.split('/')[-1].replace('-', ' ')

    date_span = soup.find('span', class_='article_byline')
    date = date_span.text.split('-')[-1].strip() if date_span else 'Unknown'

    article_body = soup.find('div', class_='article-content')
    if article_body:
        content = ' '.join(p.text for p in article_body.find_all('p'))
    else:
        content = ' '.join(p.text for p in soup.find_all('p'))

    return headline, date, content


def main(num_pages=10, output_csv=None):
    """
    Scrape crude oil news and run VADER sentiment analysis.

    Args:
        num_pages: Number of listing pages to scrape (default 10).
        output_csv: Path to save results as CSV (optional).
    """
    base_url = "https://oilprice.com"

    print(f"Scraping {num_pages} pages from {base_url}...")
    article_urls = get_article_urls(base_url, num_pages)

    print(f"Found {len(article_urls)} articles. Scraping content...")
    articles = [scrape_article_data(url) for url in article_urls]

    news_df = pd.DataFrame(articles, columns=['Headline', 'Date', 'News'])

    analyzer = SentimentIntensityAnalyzer()
    news_df['Sentiment'] = news_df['News'].apply(
        lambda text: analyzer.polarity_scores(text)['compound']
    )

    if output_csv:
        news_df.to_csv(output_csv, index=False)
        print(f"Saved to {output_csv}")

    print(news_df.head())
    return news_df


if __name__ == '__main__':
    main(num_pages=10, output_csv='data/crude_oil_news_articles.csv')
