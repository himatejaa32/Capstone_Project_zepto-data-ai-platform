import sqlite3
from pathlib import Path


# Fixed conversion rate required by the assignment
GBP_TO_INR = 105.50

# Database location
DB_PATH = Path(__file__).resolve().parent / "books.db"


def create_database():
    """
    Create the SQLite database and normalized tables.
    """

    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    # Categories table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT NOT NULL UNIQUE
        )
    """)

    # Books table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER,
            availability TEXT,
            in_stock INTEGER NOT NULL,
            category_id INTEGER NOT NULL,
            FOREIGN KEY (category_id)
                REFERENCES categories(category_id)
        )
    """)

    connection.commit()

    return connection


def insert_books(connection, books):
    """
    Insert scraped books into the normalized database.
    """

    cursor = connection.cursor()

    for book in books:

        # Insert category if it does not already exist
        cursor.execute("""
            INSERT OR IGNORE INTO categories (category_name)
            VALUES (?)
        """, (book["category"],))

        # Get category ID
        cursor.execute("""
            SELECT category_id
            FROM categories
            WHERE category_name = ?
        """, (book["category"],))

        category_id = cursor.fetchone()[0]

        # Convert GBP to INR using fixed assignment rate
        price_gbp = float(book["price_gbp"])
        price_inr = price_gbp * GBP_TO_INR

        # Insert book
        cursor.execute("""
            INSERT INTO books (
                title,
                price_gbp,
                price_inr,
                rating,
                availability,
                in_stock,
                category_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            book["title"],
            price_gbp,
            price_inr,
            book["rating"],
            book["availability"],
            int(book["in_stock"]),
            category_id
        ))

    connection.commit()