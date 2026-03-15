
## Project - IMDB 

The goal in this project is to train a binary classifier to distinguish highly rated movies from low rated movies.

Submissions for this project will be shown on the [IMDB Leaderboard](http://big-data-competitions.westeurope.cloudapp.azure.com:8080/competitions/imdb).


#### Training Data


This repository implements a preprocessing + training pipeline to build a binary classifier that predicts whether a movie is "highly rated" (label is True) or not.

Results and models are saved under `outputs/` and the leaderboard for this task is at:
[IMDB Leaderboard](http://big-data-competitions.westeurope.cloudapp.azure.com:8080/competitions/imdb)

**High level pipeline**
- Normalize raw crew JSONs (`data/raw/directing.json`, `data/raw/writing.json`) into parquet tables (`data/processed/*_clean.parquet`).
- Use DuckDB to join data and produce base feature tables (`train_features_base.parquet`, `validation_features_base.parquet`, `test_features_base.parquet`).
- Use Spark to build text features (title TF-IDF) and save sparse vectors per `tconst`.
- Train a classifier on combined numeric + title features and save model + preprocessors with `joblib`.

**Where to look in the repo**
- `src/run_pipeline.py` — orchestrates the preprocessing steps (normalize + DuckDB).
- `src/normalize_directing.py`, `src/normalize_writing.py` — normalize raw JSONs.
- `src/duckdb_pipeline.py` — builds base feature tables using DuckDB and saves parquet files in `data/processed/`.
- `src/spark_pipeline.py` — builds title TF-IDF features (HashingTF with IDF; numFeatures=2000) and returns `title_tfidf` sparse vectors.
- `src/train_combined.py` — training script that assembles numeric features and title TF-IDF into a combined sparse matrix, trains a RandomForest (optionally tuned with RandomizedSearchCV), and saves the trained model and preprocessors.

**Data files**
- `data/raw/train-*.csv` — training splits (contain `label`).
- `data/raw/validation_hidden.csv`, `data/raw/test_hidden.csv` — validation/test without labels for submission.
- `data/raw/directing.json`, `data/raw/writing.json` — crew information used to generate `num_directors` and `num_writers`.

## Features used

Numeric / engineered features (from `src/duckdb_pipeline.py` and `src/features.py`):
- `startYear`, `endYear`, `runtimeMinutes`, `numVotes` (raw numeric columns).
- `num_directors`, `num_writers` (aggregated from normalized crew tables).
- Derived features in `src/features.py` included in preprocessing pipeline:
	- `title_length`, `original_title_length`
	- `same_title` (primaryTitle == originalTitle)
	- `has_end_year` (endYear not null)
	- missing flags: `runtime_missing`, `startYear_missing`, `numVotes_missing`
	- `log_numVotes` (log1p of votes)
	- `movie_age` (current_year - startYear; current_year is 2025 in code)
	- `runtime_bucket` (bucketized runtime) and `runtime_bucket_missing`
	- `title_word_count`, `title_has_number`

Text features:
- Title TF-IDF vectors built using Spark ML pipeline: `Tokenizer` -> `HashingTF(numFeatures=2000)` -> `IDF` -> output `title_tfidf` (a sparse vector per title). These sparse vectors are converted to CSR format and horizontally stacked with numeric features for modeling.

Feature preprocessing during training (`src/train_combined.py`):
- Numeric columns get median imputation (`SimpleImputer(strategy="median")`) and scaling (`StandardScaler`).
- Title TF-IDF kept as sparse vectors (no additional scaling).

## Model(s)
- Default model: `RandomForestClassifier` (scikit-learn). Training and evaluation happen in `src/train_combined.py`.
- Hyperparameter tuning: `--tune` runs `RandomizedSearchCV` over a small RF search space (n_iter configurable).
- Trained model artifacts: saved via `joblib.dump` as a dict with keys `{"model","imputer","scaler"}` into `outputs/models/combined_rf_model.joblib` or `combined_rf_tuned_model.joblib`.

Training setup and behavior:
- The script builds an internal 80/20 stratified train/validation split.
- It evaluates three views: numeric-only, title-only, and combined, printing accuracy and classification reports for each.

## How to run
Activate the virtual environment (PowerShell):
```powershell
& .\.venv\Scripts\Activate.ps1
```

Run the preprocessing pipeline (normalizes crew files and builds DuckDB tables):
```bash
python src/run_pipeline.py
```

Train RandomForest (default 200 trees):
```bash
python src/train_combined.py --n-estimators 200 --max-depth 10
```

Run hyperparameter tuning (RandomizedSearchCV, 50 iterations):
```bash
python src/train_combined.py --tune --n-iter 50
```

Model artifacts will be written to `outputs/models/`.

## Notes and tips
- On Windows you may see Spark/Hadoop warnings about `winutils.exe` and native-hadoop libraries; these are warnings and do not block the Spark TF-IDF step in this project.
- Reaching a target validation accuracy of 0.80 will likely require additional feature engineering (e.g., TF-IDF n-grams, title preprocessing, more crew-derived features), ensembling, or larger hyperparameter/architecture searches.
- The project is structured so you can extend `src/features.py` and re-run `src/run_pipeline.py` then `src/train_combined.py`.

If you want, I can run the hyperparameter tuning, add a simple Voting ensemble, or help add more engineered features — tell me which next.
