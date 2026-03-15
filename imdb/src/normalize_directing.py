from config import RAW_DIR, PROCESSED_DIR
import pandas as pd
import json
import ast


def parse_possible_dict(value):
    """Try to parse a python literal or JSON string into a dict.

    Raises ValueError if parsing fails.
    """
    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        value = value.strip()

        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    raise ValueError(f"Could not parse dict from value: {repr(value)[:200]}")


def normalize_directing() -> None:
    """Normalize `directing.json` into a two-column parquet table (tconst, director_id)."""
    df = pd.read_json(RAW_DIR / "directing.json")

    print("Raw directing shape:", df.shape)
    print("Raw columns:", df.columns.tolist())

    # Case 1, already row-based
    if len(df) > 1 and "movie" in df.columns and "director" in df.columns:
        clean = df.rename(columns={"movie": "tconst", "director": "director_id"})
        clean = clean[["tconst", "director_id"]].copy()

    # Case 2, one-row nested dictionary format
    elif len(df) == 1 and "movie" in df.columns and "director" in df.columns:
        movie_dict = parse_possible_dict(df["movie"].iloc[0])
        director_dict = parse_possible_dict(df["director"].iloc[0])

        if set(movie_dict.keys()) != set(director_dict.keys()):
            raise ValueError("Movie keys and director keys do not match.")

        rows = []
        for key in movie_dict:
            rows.append({
                "tconst": movie_dict[key],
                "director_id": director_dict[key]
            })

        clean = pd.DataFrame(rows)

    else:
        raise ValueError("Unexpected directing.json structure")

    print("\nNormalized directing preview:")
    print(clean.head())
    print("\nNormalized directing shape:", clean.shape)

    out_path = PROCESSED_DIR / "directing_clean.parquet"
    clean.to_parquet(out_path, index=False)

    print(f"\nSaved cleaned directing table to: {out_path}")


if __name__ == "__main__":
    normalize_directing()