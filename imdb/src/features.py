import pandas as pd
import numpy as np


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["title_length"] = df["primaryTitle"].fillna("").str.len()
    df["original_title_length"] = df["originalTitle"].fillna("").str.len()

    df["same_title"] = (
        df["primaryTitle"].fillna("").str.lower()
        == df["originalTitle"].fillna("").str.lower()
    ).astype(int)

    df["has_end_year"] = df["endYear"].notna().astype(int)

    df["runtime_missing"] = df["runtimeMinutes"].isna().astype(int)
    df["startYear_missing"] = df["startYear"].isna().astype(int)
    df["numVotes_missing"] = df["numVotes"].isna().astype(int)

    df["log_numVotes"] = np.log1p(df["numVotes"].fillna(0))

    current_year = 2025

    df["movie_age"] = current_year - df["startYear"]

    df["runtime_bucket"] = pd.cut(
        df["runtimeMinutes"],
        bins=[0, 60, 90, 120, 150, 300],
        labels=[0, 1, 2, 3, 4]
    ).astype("float")

    df["runtime_bucket_missing"] = df["runtime_bucket"].isna().astype(int)

    df["title_word_count"] = (
        df["primaryTitle"]
        .fillna("")
        .str.split()
        .apply(len)
    )

    df["title_has_number"] = df["primaryTitle"].str.contains(r"\d", regex=True, na=False).astype(int)
    return df