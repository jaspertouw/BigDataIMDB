"""Run the preprocessing pipeline.

By default this will normalize directing and writing JSONs and then build
the DuckDB feature tables. Use CLI flags to skip steps for faster iteration.
"""

from typing import Sequence
import argparse
import sys

from normalize_directing import normalize_directing
from normalize_writing import normalize_writing
from duckdb_pipeline import build_feature_tables


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline")
    parser.add_argument("--skip-normalize", action="store_true", help="Skip normalizing directing/writing JSONs")
    parser.add_argument("--skip-duckdb", action="store_true", help="Skip building DuckDB feature tables")
    args = parser.parse_args(argv)

    try:
        if not args.skip_normalize:
            print("Step 1: normalize directing")
            normalize_directing()

            print("\nStep 2: normalize writing")
            normalize_writing()
        else:
            print("Skipping normalization steps")

        if not args.skip_duckdb:
            print("\nStep 3: build DuckDB feature tables")
            build_feature_tables()
        else:
            print("Skipping DuckDB feature table build")

    except Exception as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1

    print("\nPipeline completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())