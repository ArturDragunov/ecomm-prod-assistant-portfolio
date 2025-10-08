import csv
import time
import re
import os
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By # used to select elements on the page
from selenium.webdriver.common.keys import Keys # used to automate scrolling and actions like click, send_keys, etc.
from selenium.webdriver.common.action_chains import ActionChains # used to make multiple actions in a chain
from webdriver_manager.chrome import ChromeDriverManager

# Ecommerce websites scraper
class EcommerceScraper:
    def __init__(self, output_dir="data", platform="alza"):
        self.output_dir = output_dir
        self.platform = platform.lower()
        # create directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Platform configurations
        self.platform_configs = {
            "alza": {
                "base_url": "https://www.alza.cz",
                "search_url": "https://www.alza.cz/search.htm?exps={query}",
                "product_selector": "div.browsingitem",
                "title_selector": "a.name",
                "price_selector": ".price",
                "rating_selector": "div.stars",
                "reviews_selector": "span.review-count",
                "link_selector": "a.name"
            },
            "amazon_de": {
                "base_url": "https://www.amazon.de",
                "search_url": "https://www.amazon.de/s?k={query}",
                "product_selector": "div[data-component-type='s-search-result']",
                "title_selector": "h2 span",
                "price_selector": "span.a-price-whole",
                "rating_selector": "span.a-icon-alt",
                "reviews_selector": "span.a-size-base",
                "link_selector": "a"
            },
            "flipkart": {
                "base_url": "https://www.flipkart.com",
                "search_url": "https://www.flipkart.com/search?q={query}",
                "product_selector": "div[data-id]",
                "title_selector": "div.KzDlHZ",
                "price_selector": "div.Nx9bqj",
                "rating_selector": "div.XQDdHH",
                "reviews_selector": "span.Wphh3N",
                "link_selector": "a[href*='/p/']"
            }
        }

    def get_top_reviews(self, product_url, count=2):
        """Get the top reviews for a product based on platform."""
        # config = self.platform_configs[self.platform]
        
        options = Options()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-web-security")
        options.add_argument("--disable-features=VizDisplayCompositor")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        
        if not product_url.startswith("http"):
            driver.quit()
            return "No reviews found"

        try:
            driver.get(product_url)
            time.sleep(4)
            
            # Close popups (platform-specific)
            self._close_popups(driver)
            
            # Scroll to load reviews
            for _ in range(4):
                ActionChains(driver).send_keys(Keys.END).perform()
                time.sleep(1.5)

            soup = BeautifulSoup(driver.page_source, "html.parser")
            reviews = self._extract_reviews_by_platform(soup, count)
            
        except Exception as e:
            print(f"Error getting reviews: {e}")
            reviews = []

        driver.quit()
        return " || ".join(reviews) if reviews else "No reviews found"
    
    def _close_popups(self, driver):
        """Close platform-specific popups."""
        try:
            if self.platform == "alza":
                driver.find_element(By.XPATH, "//button[contains(@class, 'close')]").click()
            elif self.platform == "amazon_de":
                driver.find_element(By.XPATH, "//button[contains(@class, 'close')]").click()
            elif self.platform == "flipkart":
                driver.find_element(By.XPATH, "//button[contains(text(), '✕')]").click()
            time.sleep(1)
        except Exception:
            pass  # Popup might not exist
    
    def _extract_reviews_by_platform(self, soup, count):
        """Extract reviews based on platform-specific selectors."""
        review_selectors = {
            "alza": ["div.review-item", "div.review-text", "div.comment"],
            "amazon_de": ["div[data-hook='review-body']", "span[data-hook='review-body']", "div.review-text"],
            "flipkart": ["div._27M-vq", "div.col.EPCmJX", "div._6K-7Co"]
        }
        
        selectors = review_selectors.get(self.platform, ["div.review"])
        seen = set()
        reviews = []
        
        for selector in selectors:
            review_blocks = soup.select(selector)
            for block in review_blocks:
                text = block.get_text(separator=" ", strip=True)
                if text and text not in seen and len(text) > 10:
                    reviews.append(text)
                    seen.add(text)
                if len(reviews) >= count:
                    break
            if len(reviews) >= count:
                break
                
        return reviews
    
    def scrape_products(self, query, max_products=1, review_count=2):
        """Scrape products based on platform and search query."""
        config = self.platform_configs[self.platform]
        
        options = Options()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-web-security")
        options.add_argument("--disable-features=VizDisplayCompositor")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        search_url = config["search_url"].format(query=query.replace(' ', '+'))
        driver.get(search_url)
        time.sleep(4)

        # Close popups
        self._close_popups(driver)
        time.sleep(2)
        
        products = []
        items = driver.find_elements(By.CSS_SELECTOR, config["product_selector"])
        
        # Filter out sponsored products for Amazon
        if self.platform == "amazon_de":
            items = [item for item in items if "Sponsored" not in item.text]
        
        items = items[:max_products]
        
        for item in items:
            try:
                product_data = self._extract_product_data(item, config)
                if product_data:
                    top_reviews = self.get_top_reviews(product_data["link"], count=review_count)
                    products.append([
                        product_data["id"], 
                        product_data["title"], 
                        product_data["rating"], 
                        product_data["total_reviews"], 
                        product_data["price"], 
                        top_reviews
                    ])
            except Exception as e:
                print(f"Error occurred while processing item: {e}")
                continue

        driver.quit()
        return products
    
    def _extract_product_data(self, item, config):
        """Extract product data based on platform configuration."""
        try:
            title = item.find_element(By.CSS_SELECTOR, config["title_selector"]).text.strip()
            price = item.find_element(By.CSS_SELECTOR, config["price_selector"]).text.strip()
            
            # Try to get rating and reviews (might not exist for all products)
            try:
                rating = item.find_element(By.CSS_SELECTOR, config["rating_selector"]).text.strip()
            except:
                rating = "N/A"
                
            try:
                reviews_text = item.find_element(By.CSS_SELECTOR, config["reviews_selector"]).text.strip()
                match = re.search(r"\d+(,\d+)?", reviews_text)
                total_reviews = match.group(0) if match else "N/A"
            except:
                total_reviews = "N/A"

            # Get product link
            link_el = item.find_element(By.CSS_SELECTOR, config["link_selector"])
            href = link_el.get_attribute("href")
            product_link = href if href.startswith("http") else config["base_url"] + href
            
            # Extract product ID from URL
            product_id = self._extract_product_id(href, config["base_url"])
            
            return {
                "id": product_id,
                "title": title,
                "rating": rating,
                "total_reviews": total_reviews,
                "price": price,
                "link": product_link
            }
        except Exception as e:
            print(f"Error extracting product data: {e}")
            return None
    
    def _extract_product_id(self, href, base_url):
        """Extract product ID from URL based on platform."""
        try:
            if self.platform == "alza":
                match = re.findall(r"/([^/]+)\.htm", href)
                return match[0] if match else "N/A"
            elif self.platform == "amazon_de":
                match = re.findall(r"/dp/([^/]+)", href)
                return match[0] if match else "N/A"
            elif self.platform == "flipkart":
                match = re.findall(r"/p/(itm[0-9A-Za-z]+)", href)
                return match[0] if match else "N/A"
            else:
                return "N/A"
        except:
            return "N/A"
    
    def save_to_csv(self, data, filename="product_reviews.csv"):
        """Save the scraped product reviews to a CSV file."""
        if os.path.isabs(filename):
            path = filename
        elif os.path.dirname(filename):  # filename includes subfolder like 'data/product_reviews.csv'
            path = filename
            os.makedirs(os.path.dirname(path), exist_ok=True)
        else:
            # plain filename like 'output.csv'
            path = os.path.join(self.output_dir, filename)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["product_id", "product_title", "rating", "total_reviews", "price", "top_reviews"])
            writer.writerows(data)
        