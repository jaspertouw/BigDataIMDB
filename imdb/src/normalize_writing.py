
from config import RAW_DIR, PROCESSED_DIR
import pandas as pd


def normalize_writing() -> None:
    """Read raw `writing.json`, normalize columns and save to parquet."""
    df = pd.read_json(RAW_DIR / "writing.json")

    print("Raw writing shape:", df.shape)
    print("Raw columns:", df.columns.tolist())

    clean = df.rename(columns={"movie": "tconst", "writer": "writer_id"})
    clean = clean[["tconst", "writer_id"]].copy()

    print("\nNormalized writing preview:")
    print(clean.head())
    print("\nNormalized writing shape:", clean.shape)

    out_path = PROCESSED_DIR / "writing_clean.parquet"
    clean.to_parquet(out_path, index=False)

    print(f"\nSaved cleaned writing table to: {out_path}")


if __name__ == "__main__":
    normalize_writing()