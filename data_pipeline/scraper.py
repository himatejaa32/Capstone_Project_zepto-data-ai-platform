import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


BASE_URL = "https://books.toscrape.com/"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}


def get_soup(url):
    response = requests.get(url, headers=HEADERS, timeout=30)

    response.raise_for_status()

    # Fix encoding problems such as Â£
    response.encoding = response.apparent_encoding

    return BeautifulSoup(response.text, "html.parser")


def parse_book(card, category):
    """
    Parse one book card from a category page.
    """

    # Title
    title_element = card.select_one("h3 a")

    if not title_element:
        raise ValueError("Title not found")

    title = title_element.get("title", "").strip()

    if not title:
        raise ValueError("Empty title")

    # Price
    price_element = card.select_one("p.price_color")

    if not price_element:
        raise ValueError("Price not found")

    price_text = price_element.get_text(strip=True)

    # Remove encoding/currency problems
    price_text = (price_text.replace("Â", "").replace("£", "").strip())

    # Extract numeric value safely
    match = re.search(r"\d+(?:\.\d+)?", price_text)

    if not match:
        raise ValueError(f"Unable to parse price: {price_text}")

    price_gbp = float(match.group())

    # Rating
    rating_element = card.select_one("p.star-rating")

    if not rating_element:
        raise ValueError("Rating not found")

    rating_classes = rating_element.get("class", [])

    rating = None

    for rating_name, rating_value in RATING_MAP.items():
        if rating_name in rating_classes:
            rating = rating_value
            break

    if rating is None:
        raise ValueError(
            f"Unable to parse rating: {rating_classes}"
        )

    # Availability
    availability_element = card.select_one("p.instock.availability")

    availability = (availability_element.get_text(" ", strip=True)
        if availability_element
        else ""
)

    in_stock = "In stock" in availability

    return {
        "title": title,
        "price_gbp": price_gbp,
        "rating": rating,
        "availability": availability,
        "in_stock": in_stock,
        "category": category
    }


def scrape_category(category_url, category_name):
    """
    Scrape all pages for one category.
    """

    books = []

    current_url = category_url

    while current_url:

        print(f"Scraping {current_url}")

        try:
            soup = get_soup(current_url)

        except Exception as error:
            print(f"Error loading page: {error}")
            break

        book_cards = soup.select("article.product_pod")

        for card in book_cards:

            try:
                book = parse_book(
                    card,
                    category_name
                )

                books.append(book)

            except Exception as error:
                print(
                    f"Skipping Book because of Error parsing book: {error}"
                )

        # Find next page
        next_link = soup.select_one(
            "li.next a"
        )

        if next_link:

            current_url = urljoin(
                current_url,
                next_link.get("href")
            )

        else:
            current_url = None

    return books


def get_categories():
    """
    Get all book categories from the main page.
    """

    soup = get_soup(BASE_URL)

    categories = []

    for link in soup.select(
        "div.side_categories ul li ul li a"
    ):

        category_name = link.get_text(
            strip=True
        )

        category_url = urljoin(
            BASE_URL,
            link.get("href")
        )

        categories.append(
            {
                "name": category_name,
                "url": category_url
            }
        )

    return categories


def scrape_books():
    """
    Scrape books from all available categories.
    """

    all_books = []

    categories = get_categories()

    print(
        f"Found {len(categories)} categories."
    )

    for category in categories:

        category_books = scrape_category(
            category["url"],
            category["name"]
        )

        all_books.extend(category_books)

        print(
            f"Category '{category['name']}' "
            f"scraped {len(category_books)} books."
        )

    return all_books

