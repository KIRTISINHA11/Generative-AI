"""
WEB SCRAPER FOR PRODUCT REVIEWS using BeautifulSoup
Scrapes real reviews from product pages
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from typing import List

# Headers to mimic real browser
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

class BeautifulSoupScraper:
    """Scrapes product reviews using BeautifulSoup"""

    def __init__(self):
        self.reviews = []
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def scrape_amazon_reviews(self, asin: str, num_reviews: int = 100) -> List[str]:
        """
        Scrape reviews from Amazon using ASIN (product ID)

        Example ASIN: B09XXXXX (from URL: amazon.com/dp/B09XXXXX)
        Note: Amazon heavily blocks scraping. This may not work for all products.
        """
        print(f"🔄 Attempting to scrape {num_reviews} reviews from Amazon...")

        reviews = []

        # Amazon reviews page URL
        base_url = f"https://www.amazon.com/gp/customer-reviews/{asin}/"

        try:
            response = self.session.get(base_url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Find all review divs
            review_divs = soup.find_all('div', {'data-hook': 'review'})

            print(f"Found {len(review_divs)} reviews on page")

            for review_div in review_divs[:num_reviews]:
                # Extract review text
                review_text_elem = review_div.find('span', {'data-hook': 'review-body'})

                if review_text_elem:
                    review_text = review_text_elem.get_text(strip=True)
                    if len(review_text) > 10:  # Only keep substantial reviews
                        reviews.append(review_text)
                        print(f"✓ Review {len(reviews)}: {review_text[:50]}...")

            print(f"✅ Successfully scraped {len(reviews)} reviews")
            return reviews

        except requests.exceptions.RequestException as e:
            print(f"❌ Error: {e}")
            print("💡 Amazon blocks automated scraping. Try with different ASIN or use sample data.")
            return []

    def scrape_flipkart_reviews(self, product_url: str, num_reviews: int = 100) -> List[str]:
        """
        Scrape reviews from Flipkart using BeautifulSoup

        Example: https://www.flipkart.com/product-name/p/itmxxxxxxxx
        """
        print(f"🔄 Attempting to scrape {num_reviews} reviews from Flipkart...")

        reviews = []

        try:
            response = self.session.get(product_url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Find review elements (adjust selectors based on Flipkart's structure)
            review_divs = soup.find_all('div', {'class': '_3DCqkx'})

            print(f"Found {len(review_divs)} reviews on page")

            for review_div in review_divs[:num_reviews]:
                # Extract review text
                review_text_elem = review_div.find('div', {'class': '_16PfRd'})

                if review_text_elem:
                    review_text = review_text_elem.get_text(strip=True)
                    if len(review_text) > 10:
                        reviews.append(review_text)
                        print(f"✓ Review {len(reviews)}: {review_text[:50]}...")

            print(f"✅ Successfully scraped {len(reviews)} reviews")
            return reviews

        except requests.exceptions.RequestException as e:
            print(f"❌ Error: {e}")
            print("💡 Flipkart may be blocking scraping. Try with different URL or use sample data.")
            return []

    def scrape_generic_site(self, url: str, review_selector: str, num_reviews: int = 100) -> List[str]:
        """
        Generic scraper for custom websites

        Args:
            url: Website URL
            review_selector: CSS selector for review text (e.g., '.review-text', 'div.comment')
            num_reviews: Number of reviews to scrape
        """
        print(f"🔄 Scraping from: {url}")

        reviews = []

        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Find review elements using CSS selector
            review_elements = soup.select(review_selector)

            print(f"Found {len(review_elements)} potential reviews")

            for elem in review_elements[:num_reviews]:
                review_text = elem.get_text(strip=True)
                if len(review_text) > 10:  # Only keep substantial reviews
                    reviews.append(review_text)
                    print(f"✓ Review {len(reviews)}: {review_text[:50]}...")

            print(f"✅ Successfully scraped {len(reviews)} reviews")
            return reviews

        except Exception as e:
            print(f"❌ Error: {e}")
            return []

    def create_realistic_reviews(self) -> List[str]:
        """
        Create realistic product reviews for demonstration
        (Use when live scraping is blocked)
        """
        print("📝 Creating 100 realistic product reviews for demonstration...")

        reviews = [
            "This product is absolutely amazing! Best purchase I've made in years. Highly recommend to everyone.",
            "Terrible quality. Broke after one week. Complete waste of money. Do not buy this!",
            "Excellent product! Exceeded all my expectations. Perfect quality and fast delivery.",
            "Poor quality. Not as described in the listing. Very disappointed with this purchase.",
            "Fantastic! Works exactly as advertised. This is the best product ever!",
            "Bad experience overall. Item arrived damaged. Customer service was unhelpful.",
            "Love it! Great craftsmanship and very durable. Will definitely buy again.",
            "Horrible product. Failed immediately. Total waste of money and time.",
            "Perfect! Exactly what I needed. Excellent quality and amazing customer service.",
            "Not recommended. Found many defects. Quality is very poor overall.",
            "Five stars! Everything is perfect about this product. Highly recommended!",
            "Waste of money. Low quality material. Should have read reviews first.",
            "Very satisfied! Great value for money. Excellent service and quality.",
            "Not worth buying. Multiple issues found. Quality could be much better.",
            "Amazing! Best product in this category. Definitely worth the price.",
            "Disappointing. Not as good as other brands. Expected better quality.",
            "Excellent! Great quality and fast delivery. Very happy with purchase.",
            "Very bad. Product stopped working after few days. Refund was difficult.",
            "Perfect for everyday use. Great quality at affordable price. Recommended!",
            "Terrible quality. Poor packaging. Item arrived in bad condition.",
            "Outstanding! Best investment ever. Works perfectly. Five stars always!",
            "Poor experience. Customer service didn't help. Very unsatisfied customer.",
            "Great! Amazing quality. Fast shipping. Exactly as described. Perfect!",
            "Bad quality. Too expensive for what you get. Very disappointed overall.",
            "Excellent! Great build quality. Worth every penny. Highly recommended!",
            "Horrible experience. Product broken on arrival. Terrible quality overall.",
            "Perfect purchase! Amazing value. Great quality. Will buy again soon.",
            "Not good. Product has many defects. Below expectations completely.",
            "Fantastic! Love everything about it. Best product ever. Five stars!",
            "Waste of money. Poor quality. Avoid this product completely.",
            "Amazing! Exactly as advertised. Great quality throughout. Very satisfied.",
            "Bad quality. Not worth the price. Many problems with this product.",
            "Excellent! Great customer service. Fast delivery. Perfect product overall.",
            "Disappointing. Poor quality overall. Not recommended by me at all.",
            "Perfect! Amazing quality. Great value for money. Highly recommended!",
            "Terrible. Product doesn't work properly. Very bad experience overall.",
            "Great product! Perfect quality. Fast shipping. Very happy customer.",
            "Poor quality. Overpriced. Not satisfied with my purchase at all.",
            "Fantastic! Best product in the market. Excellent quality. Recommended!",
            "Not recommended. Low quality throughout. Had to return the product.",
            "Excellent! Great product. Perfect for my needs. Very satisfied overall.",
            "Bad experience. Product broke quickly. Poor customer service team.",
            "Amazing! Love the quality. Great value. Will buy again very soon.",
            "Disappointing quality. Not as advertised. Very unsatisfied overall.",
            "Perfect! Excellent quality. Fast delivery. Highly recommended always!",
            "Horrible. Complete waste of money. Very bad quality throughout.",
            "Great! Perfect product. Amazing quality. Very happy customer overall.",
            "Poor quality. Not good value for money. Not recommended at all.",
            "Fantastic! Best purchase ever. Excellent quality. Five stars always!",
            "Not good. Has many defects. Quality is poor. Avoid completely.",
            "Excellent! Perfect quality. Great value. Highly recommend always!",
            "Terrible product. Not worth buying. Waste of time and money.",
            "Perfect! Best quality ever. Great price. Will buy again soon.",
            "Bad experience. Poor quality. Many issues found. Not recommended.",
            "Amazing product! Best ever made. Excellent value for money.",
            "Disappointing. Low quality. Not as described. Very unsatisfied.",
            "Excellent! Worth every penny. Great quality. Highly recommended!",
            "Horrible. Product failed. Poor quality. Very disappointed overall.",
            "Great quality! Perfect for use. Fast service. Very happy.",
            "Poor quality. Too expensive. Not satisfied with purchase.",
            "Fantastic product! Best quality. Excellent value. Recommended!",
            "Not good. Has defects. Quality poor. Don't recommend.",
            "Excellent! Perfect quality. Great value. Recommend highly!",
            "Bad quality. Not worth it. Disappointed completely.",
            "Amazing! Best product ever. Great quality. Perfect!",
            "Disappointing. Poor quality. Not recommended at all.",
            "Perfect! Excellent quality. Highly recommended always!",
            "Horrible. Waste of money. Bad quality overall.",
            "Great! Perfect quality. Very happy customer overall.",
            "Poor quality. Not good value. Not recommended.",
            "Fantastic! Best product. Excellent quality. Recommended!",
            "Not good. Many defects. Quality poor.",
            "Excellent! Perfect quality. Great value.",
            "Terrible. Not worth buying. Bad quality.",
            "Perfect! Best quality. Great price.",
            "Bad experience. Poor quality. Issues found.",
            "Amazing! Best ever. Excellent value.",
            "Disappointing. Low quality. Unsatisfied.",
            "Excellent! Worth it. Great quality.",
            "Horrible. Failed. Very disappointed.",
            "Great! Perfect. Very happy.",
            "Poor. Not satisfied. Disappointed.",
            "Fantastic! Best. Recommended.",
            "Not good. Has issues. Poor.",
            "Excellent! Perfect. Highly recommend.",
            "Bad. Not worth it. Disappointed.",
            "Amazing! Best. Great.",
            "Disappointing. Poor. Not good.",
            "Perfect! Excellent. Recommended.",
            "Horrible. Bad. Waste.",
            "Great! Perfect. Happy.",
            "Poor. Not good. Disappointed.",
            "Fantastic! Best. Great.",
            "Not good. Bad. Issues.",
            "Excellent! Perfect. Great.",
            "Terrible. Bad. Not worth.",
            "Perfect! Best. Great.",
            "Bad. Poor. Disappointed.",
            "Amazing! Excellent. Best.",
        ]

        print(f"✓ Created {len(reviews)} realistic product reviews")
        return reviews


def main():
    """Main scraper function"""

    print("\n" + "="*70)
    print("PRODUCT REVIEW SCRAPER - BeautifulSoup Edition")
    print("="*70)

    scraper = BeautifulSoupScraper()
    reviews = []

    print("\n📌 SCRAPING OPTIONS:")
    print("1️⃣  Try to scrape from Amazon")
    print("2️⃣  Try to scrape from Flipkart")
    print("3️⃣  Scrape from custom website (you provide URL and CSS selector)")
    print("4️⃣  Use realistic sample reviews (recommended if above blocked)")

    choice = input("\nSelect option (1-4): ").strip()

    if choice == '1':
        print("\n📚 Amazon Scraping Instructions:")
        print("  1. Go to Amazon product page")
        print("  2. Copy the ASIN from URL: /dp/[ASIN]")
        print("  Example: amazon.com/dp/B09ABC123XYZ → ASIN = B09ABC123XYZ")
        asin = input("\nEnter Amazon ASIN: ").strip()
        if asin:
            reviews = scraper.scrape_amazon_reviews(asin, num_reviews=100)
        else:
            print("❌ No ASIN provided. Using sample reviews...")
            reviews = scraper.create_realistic_reviews()

    elif choice == '2':
        print("\n📚 Flipkart Scraping Instructions:")
        print("  1. Go to Flipkart product page")
        print("  2. Copy the full URL")
        url = input("\nEnter Flipkart product URL: ").strip()
        if url:
            reviews = scraper.scrape_flipkart_reviews(url, num_reviews=100)
        else:
            print("❌ No URL provided. Using sample reviews...")
            reviews = scraper.create_realistic_reviews()

    elif choice == '3':
        print("\n📚 Custom Website Scraping:")
        print("  Example CSS selectors:")
        print("    - '.review-text' (class selector)")
        print("    - 'div.comment' (element with class)")
        print("    - '[data-hook=review-body]' (attribute selector)")

        url = input("\nEnter website URL: ").strip()
        selector = input("Enter CSS selector for review text: ").strip()

        if url and selector:
            reviews = scraper.scrape_generic_site(url, selector, num_reviews=100)
        else:
            print("❌ Missing URL or selector. Using sample reviews...")
            reviews = scraper.create_realistic_reviews()

    else:
        # Option 4: Use sample reviews
        reviews = scraper.create_realistic_reviews()

    # If scraping failed, fallback to samples
    if not reviews:
        print("\n⚠️ Scraping failed or returned no results.")
        print("💡 This is normal - most e-commerce sites block automated scraping.")
        print("📝 Using realistic sample reviews instead...\n")
        reviews = scraper.create_realistic_reviews()

    # Save to CSV
    df = pd.DataFrame({'review_text': reviews})
    df.to_csv('reviews.csv', index=False)

    print("\n" + "="*70)
    print("✅ DATA SAVED")
    print("="*70)
    print(f"\n✓ Saved {len(reviews)} reviews to: reviews.csv")
    print(f"\nDataset Statistics:")
    print(f"  • Total reviews: {len(reviews)}")
    print(f"  • Average length: {df['review_text'].str.len().mean():.0f} characters")
    print(f"  • Max length: {df['review_text'].str.len().max()} characters")
    print(f"  • Min length: {df['review_text'].str.len().min()} characters")

    print(f"\nFirst 5 reviews:")
    for i, review in enumerate(reviews[:5], 1):
        print(f"  {i}. {review}")

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n✓ Run text feature engineering:")
    print("  python text_feature_engineering.py\n")

    return df


if __name__ == "__main__":
    df = main()
