from scraper import scrape_books
from database import create_database, insert_books


def main():
    print("Starting the data pipeline...")

    books = scrape_books()

    print(f"Scraped {len(books)} books.")

    if len(books) < 60:
        raise ValueError(
            "Scraped less than 60 books. "
            "Aborting the pipeline."
        )

    connection = create_database()

    insert_books(
        connection,
        books
    )

    connection.close()

    print("Data pipeline completed successfully.")


if __name__ == "__main__":
    main()