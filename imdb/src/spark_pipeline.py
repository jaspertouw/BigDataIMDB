from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml import Pipeline

from config import PROCESSED_DIR


def build_title_features():
    spark = SparkSession.builder \
        .appName("IMDB Title Features") \
        .master("local[*]") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    train = spark.read.parquet(str(PROCESSED_DIR / "train_features_base.parquet"))
    validation = spark.read.parquet(str(PROCESSED_DIR / "validation_features_base.parquet"))
    test = spark.read.parquet(str(PROCESSED_DIR / "test_features_base.parquet"))

    tokenizer = Tokenizer(inputCol="primaryTitle", outputCol="title_tokens")
    hashing = HashingTF(inputCol="title_tokens", outputCol="title_tf", numFeatures=2000)
    idf = IDF(inputCol="title_tf", outputCol="title_tfidf")

    pipeline = Pipeline(stages=[tokenizer, hashing, idf])
    model = pipeline.fit(train)

    train_features = model.transform(train).select("tconst", "title_tfidf")
    validation_features = model.transform(validation).select("tconst", "title_tfidf")
    test_features = model.transform(test).select("tconst", "title_tfidf")

    print("\nTrain title rows:", train_features.count())
    print("Validation title rows:", validation_features.count())
    print("Test title rows:", test_features.count())

    print("\nExample sparse title features:")
    train_features.show(5, truncate=False)

    return spark, train_features, validation_features, test_features


if __name__ == "__main__":
    spark, train_features, validation_features, test_features = build_title_features()
    spark.stop()