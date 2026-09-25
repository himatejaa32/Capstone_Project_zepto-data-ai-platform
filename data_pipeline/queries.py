import sqlite3
from pathlib import Path

import pandas as pd


DB_PATH = Path(__file__).resolve().parent / "books.db"


def run_query(connection, title, query):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    df = pd.read_sql_query(query, connection)

    print(df.to_string(index=False))

    return df


def main():

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}\n"
            "Run data_pipeline/main.py first."
        )

    connection = sqlite3.connect(DB_PATH)

    # ---------------------------------------------------------
    # Query 1 - SELECT + WHERE
    # ---------------------------------------------------------
    run_query(
        connection,
        "QUERY 1 - Books with rating >= 4",
        """
        SELECT
            title,
            price_gbp,
            rating
        FROM books
        WHERE rating >= 4
        ORDER BY rating DESC
        LIMIT 10;
        """
    )

    # ---------------------------------------------------------
    # Query 2 - ORDER BY
    # ---------------------------------------------------------
    run_query(
        connection,
        "QUERY 2 - Most expensive books",
        """
        SELECT
            title,
            price_gbp
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 10;
        """
    )

    # ---------------------------------------------------------
    # Query 3 - LIMIT
    # ---------------------------------------------------------
    run_query(
        connection,
        "QUERY 3 - First 5 books",
        """
        SELECT
            book_id,
            title,
            price_gbp,
            rating
        FROM books
        LIMIT 5;
        """
    )

    # ---------------------------------------------------------
    # Query 4 - DISTINCT
    # ---------------------------------------------------------
    run_query(
        connection,
        "QUERY 4 - Distinct categories",
        """
        SELECT DISTINCT category_name
        FROM categories
        ORDER BY category_name;
        """
    )

    # ---------------------------------------------------------
    # Query 5 - IN
    # ---------------------------------------------------------
    run_query(
        connection,
        "QUERY 5 - Books from selected categories",
        """
        SELECT
            title,
            price_gbp,
            rating
        FROM books
        WHERE category_id IN (
            SELECT category_id
            FROM categories
            WHERE category_name IN (
                'Classics',
                'Romance',
                'Science Fiction'
            )
        )
        LIMIT 20;
        """
    )

    # ---------------------------------------------------------
    # Query 6 - BETWEEN
    # ---------------------------------------------------------
    run_query(
        connection,
        "QUERY 6 - Books priced between £10 and £30",
        """
        SELECT
            title,
            price_gbp,
            rating
        FROM books
        WHERE price_gbp BETWEEN 10 AND 30
        ORDER BY price_gbp;
        """
    )

    # ---------------------------------------------------------
    # Query 7 - JOIN
    # ---------------------------------------------------------
    join_df = run_query(
        connection,
        "QUERY 7 - Books with categories",
        """
        SELECT
            b.title,
            b.price_gbp,
            b.price_inr,
            b.rating,
            c.category_name
        FROM books b
        JOIN categories c
            ON b.category_id = c.category_id
        ORDER BY b.price_gbp DESC
        LIMIT 20;
        """
    )

    # ---------------------------------------------------------
    # pandas SQL read
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("PANDAS read_sql QUERY")
    print("=" * 70)

    pandas_df = pd.read_sql(
        """
        SELECT
            title,
            price_gbp,
            price_inr,
            rating,
            in_stock
        FROM books
        WHERE rating = 5
        LIMIT 10;
        """,
        connection
    )

    print(pandas_df.to_string(index=False))

    # ---------------------------------------------------------
    # Reproduce JOIN using pandas.merge()
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("PANDAS MERGE - Reproducing SQL JOIN")
    print("=" * 70)

    books_df = pd.read_sql(
        """
        SELECT
            book_id,
            title,
            price_gbp,
            price_inr,
            rating,
            category_id
        FROM books;
        """,
        connection
    )

    categories_df = pd.read_sql(
        """
        SELECT
            category_id,
            category_name
        FROM categories;
        """,
        connection
    )

    merged_df = pd.merge(
        books_df,
        categories_df,
        on="category_id",
        how="inner"
    )

    print(
        merged_df[
            [
                "title",
                "price_gbp",
                "price_inr",
                "rating",
                "category_name"
            ]
        ].head(20).to_string(index=False)
    )

    connection.close()

    print("\nSQL queries completed successfully.")


if __name__ == "__main__":
    main()