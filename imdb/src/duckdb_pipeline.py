import duckdb
from config import RAW_DIR, PROCESSED_DIR


def build_feature_tables():
    con = duckdb.connect()

    train_pattern = str(RAW_DIR / "train-*.csv")
    validation_path = str(RAW_DIR / "validation_hidden.csv")
    test_path = str(RAW_DIR / "test_hidden.csv")

    directing_clean_path = str(PROCESSED_DIR / "directing_clean.parquet")
    writing_clean_path = str(PROCESSED_DIR / "writing_clean.parquet")

    # Main CSV tables
    con.execute(f"""
        CREATE OR REPLACE TABLE train_raw AS
        SELECT *
        FROM read_csv_auto('{train_pattern}', union_by_name=True, header=True);
    """)

    con.execute(f"""
        CREATE OR REPLACE TABLE validation_raw AS
        SELECT *
        FROM read_csv_auto('{validation_path}', header=True);
    """)

    con.execute(f"""
        CREATE OR REPLACE TABLE test_raw AS
        SELECT *
        FROM read_csv_auto('{test_path}', header=True);
    """)

    # Clean main tables
    con.execute("""
        CREATE OR REPLACE TABLE train_clean AS
        SELECT
            CAST(tconst AS VARCHAR) AS tconst,
            CAST(primaryTitle AS VARCHAR) AS primaryTitle,
            CAST(originalTitle AS VARCHAR) AS originalTitle,
            TRY_CAST(startYear AS DOUBLE) AS startYear,
            TRY_CAST(endYear AS DOUBLE) AS endYear,
            TRY_CAST(runtimeMinutes AS DOUBLE) AS runtimeMinutes,
            TRY_CAST(numVotes AS DOUBLE) AS numVotes,
            CAST(label AS BOOLEAN) AS label
        FROM train_raw;
    """)

    con.execute("""
        CREATE OR REPLACE TABLE validation_clean AS
        SELECT
            CAST(tconst AS VARCHAR) AS tconst,
            CAST(primaryTitle AS VARCHAR) AS primaryTitle,
            CAST(originalTitle AS VARCHAR) AS originalTitle,
            TRY_CAST(startYear AS DOUBLE) AS startYear,
            TRY_CAST(endYear AS DOUBLE) AS endYear,
            TRY_CAST(runtimeMinutes AS DOUBLE) AS runtimeMinutes,
            TRY_CAST(numVotes AS DOUBLE) AS numVotes
        FROM validation_raw;
    """)

    con.execute("""
        CREATE OR REPLACE TABLE test_clean AS
        SELECT
            CAST(tconst AS VARCHAR) AS tconst,
            CAST(primaryTitle AS VARCHAR) AS primaryTitle,
            CAST(originalTitle AS VARCHAR) AS originalTitle,
            TRY_CAST(startYear AS DOUBLE) AS startYear,
            TRY_CAST(endYear AS DOUBLE) AS endYear,
            TRY_CAST(runtimeMinutes AS DOUBLE) AS runtimeMinutes,
            TRY_CAST(numVotes AS DOUBLE) AS numVotes
        FROM test_raw;
    """)

    # Read normalized crew tables
    con.execute(f"""
        CREATE OR REPLACE TABLE directing_clean AS
        SELECT * FROM read_parquet('{directing_clean_path}');
    """)

    con.execute(f"""
        CREATE OR REPLACE TABLE writing_clean AS
        SELECT * FROM read_parquet('{writing_clean_path}');
    """)

    # Aggregate crew features
    con.execute("""
        CREATE OR REPLACE TABLE director_counts AS
        SELECT
            tconst,
            COUNT(*) AS num_directors
        FROM directing_clean
        GROUP BY tconst;
    """)

    con.execute("""
        CREATE OR REPLACE TABLE writer_counts AS
        SELECT
            tconst,
            COUNT(*) AS num_writers
        FROM writing_clean
        GROUP BY tconst;
    """)

    # Join onto train
    con.execute("""
        CREATE OR REPLACE TABLE train_features_base AS
        SELECT
            t.*,
            COALESCE(d.num_directors, 0) AS num_directors,
            COALESCE(w.num_writers, 0) AS num_writers
        FROM train_clean t
        LEFT JOIN director_counts d ON t.tconst = d.tconst
        LEFT JOIN writer_counts w ON t.tconst = w.tconst;
    """)

    # Join onto validation
    con.execute("""
        CREATE OR REPLACE TABLE validation_features_base AS
        SELECT
            v.*,
            COALESCE(d.num_directors, 0) AS num_directors,
            COALESCE(w.num_writers, 0) AS num_writers
        FROM validation_clean v
        LEFT JOIN director_counts d ON v.tconst = d.tconst
        LEFT JOIN writer_counts w ON v.tconst = w.tconst;
    """)

    # Join onto test
    con.execute("""
        CREATE OR REPLACE TABLE test_features_base AS
        SELECT
            t.*,
            COALESCE(d.num_directors, 0) AS num_directors,
            COALESCE(w.num_writers, 0) AS num_writers
        FROM test_clean t
        LEFT JOIN director_counts d ON t.tconst = d.tconst
        LEFT JOIN writer_counts w ON t.tconst = w.tconst;
    """)

    print("\nTrain rows:")
    print(con.execute("SELECT COUNT(*) FROM train_features_base").fetchone()[0])

    print("\nValidation rows:")
    print(con.execute("SELECT COUNT(*) FROM validation_features_base").fetchone()[0])

    print("\nTest rows:")
    print(con.execute("SELECT COUNT(*) FROM test_features_base").fetchone()[0])

    print("\nMovies with director info in train:")
    print(con.execute("""
        SELECT COUNT(*) FROM train_features_base WHERE num_directors > 0
    """).fetchone()[0])

    print("\nMovies with writer info in train:")
    print(con.execute("""
        SELECT COUNT(*) FROM train_features_base WHERE num_writers > 0
    """).fetchone()[0])

    print("\nTrain sample:")
    print(con.execute("SELECT * FROM train_features_base LIMIT 5").fetchdf())

    con.execute(f"""
        COPY train_features_base TO '{PROCESSED_DIR / "train_features_base.parquet"}' (FORMAT PARQUET);
    """)
    con.execute(f"""
        COPY validation_features_base TO '{PROCESSED_DIR / "validation_features_base.parquet"}' (FORMAT PARQUET);
    """)
    con.execute(f"""
        COPY test_features_base TO '{PROCESSED_DIR / "test_features_base.parquet"}' (FORMAT PARQUET);
    """)

    print("\nMissing values in train_features_base:")
    print(con.execute("""
        SELECT
            SUM(CASE WHEN startYear IS NULL THEN 1 ELSE 0 END) AS missing_startYear,
            SUM(CASE WHEN endYear IS NULL THEN 1 ELSE 0 END) AS missing_endYear,
            SUM(CASE WHEN runtimeMinutes IS NULL THEN 1 ELSE 0 END) AS missing_runtimeMinutes,
            SUM(CASE WHEN numVotes IS NULL THEN 1 ELSE 0 END) AS missing_numVotes
        FROM train_features_base
    """).fetchdf())

    print("\nRuntime distribution summary:")
    print(con.execute("""
        SELECT
            MIN(runtimeMinutes) AS min_runtime,
            MAX(runtimeMinutes) AS max_runtime,
            AVG(runtimeMinutes) AS avg_runtime
        FROM train_features_base
    """).fetchdf())

    print("\nVote distribution summary:")
    print(con.execute("""
        SELECT
            MIN(numVotes) AS min_votes,
            MAX(numVotes) AS max_votes,
            AVG(numVotes) AS avg_votes
        FROM train_features_base
    """).fetchdf())

    print("\nDirector count distribution:")
    print(con.execute("""
        SELECT
            num_directors,
            COUNT(*) AS n_movies
        FROM train_features_base
        GROUP BY num_directors
        ORDER BY num_directors
        LIMIT 15
    """).fetchdf())

    print("\nWriter count distribution:")
    print(con.execute("""
        SELECT
            num_writers,
            COUNT(*) AS n_movies
        FROM train_features_base
        GROUP BY num_writers
        ORDER BY num_writers
        LIMIT 15
    """).fetchdf())

    con.close()


if __name__ == "__main__":
    build_feature_tables()