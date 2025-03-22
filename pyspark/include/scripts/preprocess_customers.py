"""
This script preprocesses Kindle Store data using PySpark with improved efficiency.
It computes counts using distributed dictionary aggregations (to avoid large broadcast joins)
and categorises rows into store bins based on these counts. Typecasting errors are logged to log.txt,
and the output directory is safely removed before writing to avoid directory creation errors.
"""

def safe_write_partitioned(df, output_dir, spark):
    # Use Hadoop API to delete the output directory if it exists.
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
    p = spark._jvm.org.apache.hadoop.fs.Path(output_dir)
    if fs.exists(p):
        fs.delete(p, True)
    # Write the DataFrame partitioned by store_category.
    df.write.mode("overwrite").partitionBy("store_category").parquet(output_dir)

def main():
    import os, datetime
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, when, round as pyspark_round, trim, lit, min as spark_min, max as spark_max, udf
    from pyspark.sql.types import IntegerType, FloatType, StringType, BooleanType, LongType

    spark = SparkSession.builder \
        .appName("KindleStorePreprocessing") \
        .config("spark.ui.port", "7077") \
        .config("spark.driver.memory", "32g") \
        .config("spark.rpc.message.maxSize", "512") \
        .config("spark.sql.autoBroadcastJoinThreshold", "-1") \
        .config("spark.sql.parquet.enableVectorizedReader", "false") \
        .getOrCreate()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(BASE_DIR, "..", "Kindle_Store.parquet")
    df = spark.read.parquet(input_path).repartition(2).drop("images")

    try:
        df = df.withColumn("rating_clean",
            when(col("rating") < 0, lit(0))
            .when(col("rating") > 5, lit(5))
            .otherwise(pyspark_round(col("rating")))
            .cast(IntegerType())
        )
    except Exception as e:
        with open("/tmp/log.txt", "a") as log_file:
            ts = datetime.datetime.now().strftime("[%d/%m/%Y %H:%M:%S:%f]")[:-3]
            log_file.write(f"{ts} rating_clean conversion failed: {type(e).__name__}\n")

    try:
        df = df.withColumn("title_clean",
            when((col("title").isNull()) | (trim(col("title")) == ""), lit(""))
            .otherwise(col("title").cast(StringType()))
        )
    except Exception as e:
        with open("/tmp/log.txt", "a") as log_file:
            ts = datetime.datetime.now().strftime("[%d/%m/%Y %H:%M:%S:%f]")[:-3]
            log_file.write(f"{ts} title_clean conversion failed: {type(e).__name__}\n")

    try:
        df = df.withColumn("text_clean",
            when((col("text").isNull()) | (trim(col("text")) == ""), lit(""))
            .otherwise(col("text").cast(StringType()))
        )
    except Exception as e:
        with open("/tmp/log.txt", "a") as log_file:
            ts = datetime.datetime.now().strftime("[%d/%m/%Y %H:%M:%S:%f]")[:-3]
            log_file.write(f"{ts} text_clean conversion failed: {type(e).__name__}\n")

    try:
        df = df.withColumn("asin_clean", col("asin").cast(StringType())) \
               .withColumn("parent_asin_clean", col("parent_asin").cast(StringType())) \
               .withColumn("user_id_clean", col("user_id").cast(StringType())) \
               .withColumn("verified_purchase_clean", col("verified_purchase").cast(BooleanType())) \
               .withColumn("timestamp_clean", col("timestamp").cast(LongType())) \
               .withColumn("timestamp_days", (col("timestamp") / 86400000).cast(FloatType())) \
               .withColumn("helpful_vote_clean", col("helpful_vote").cast(IntegerType()))
    except Exception as e:
        with open("/tmp/log.txt", "a") as log_file:
            ts = datetime.datetime.now().strftime("[%d/%m/%Y %H:%M:%S:%f]")[:-3]
            log_file.write(f"{ts} typecasting failed: {type(e).__name__}\n")

    agg_vals = df.agg(
        spark_min("timestamp_days").alias("min_ts"),
        spark_max("timestamp_days").alias("max_ts"),
        spark_min("helpful_vote_clean").alias("min_hv"),
        spark_max("helpful_vote_clean").alias("max_hv")
    ).collect()[0]
    min_ts, max_ts, min_hv, max_hv = agg_vals["min_ts"], agg_vals["max_ts"], agg_vals["min_hv"], agg_vals["max_hv"]

    df = df.withColumn("normalized_timestamp",
            when(lit(max_ts) - lit(min_ts) == 0, lit(0.0))
            .otherwise((col("timestamp_days") - lit(min_ts)) / (lit(max_ts) - lit(min_ts)))
        )
    df = df.withColumn("normalized_helpful_vote",
            when(lit(max_hv) - lit(min_hv) == 0, lit(0.0))
            .otherwise((col("helpful_vote_clean") - lit(min_hv)) / (lit(max_hv) - lit(min_hv)))
        )

    user_counts_dict = df.select("user_id_clean").rdd.map(lambda r: (r[0], 1)).reduceByKey(lambda a, b: a + b).collectAsMap()
    asin_counts_dict = df.select("parent_asin_clean").rdd.map(lambda r: (r[0], 1)).reduceByKey(lambda a, b: a + b).collectAsMap()
    bc_user = spark.sparkContext.broadcast(user_counts_dict)
    bc_asin = spark.sparkContext.broadcast(asin_counts_dict)

    def get_user_count(uid):
        return bc_user.value.get(uid, 0)
    def get_asin_count(pid):
        return bc_asin.value.get(pid, 0)
    get_user_count_udf = udf(get_user_count, IntegerType())
    get_asin_count_udf = udf(get_asin_count, IntegerType())
    df = df.withColumn("user_count", get_user_count_udf(col("user_id_clean"))) \
           .withColumn("asin_count", get_asin_count_udf(col("parent_asin_clean")))

    # Categorise rows based on count thresholds
    df = df.withColumn("store_category",
            when((col("user_count") > 20) & (col("asin_count") > 20), lit("Kindle_Store_20"))
            .when((col("user_count") > 15) & (col("asin_count") > 15), lit("Kindle_Store_15"))
            .when((col("user_count") > 10) & (col("asin_count") > 10), lit("Kindle_Store_10"))
            .when((col("user_count") > 5) & (col("asin_count") > 5), lit("Kindle_Store_5"))
            .when((col("user_count") > 4) & (col("asin_count") > 4), lit("Kindle_Store_4"))
            .when((col("user_count") > 3) & (col("asin_count") > 3), lit("Kindle_Store_3"))
            .when((col("user_count") > 2) & (col("asin_count") > 2), lit("Kindle_Store_2"))
            .when((col("user_count") > 1) & (col("asin_count") > 1), lit("Kindle_Store_1"))
            .otherwise(lit("Kindle_Store_0"))
        )

    final_df = df.select(
        col("rating_clean").alias("rating"),
        col("title_clean").alias("title"),
        col("text_clean").alias("text"),
        col("asin_clean").alias("asin"),
        col("parent_asin_clean").alias("parent_asin"),
        col("user_id_clean").alias("user_id"),
        col("normalized_timestamp").alias("normalised_timestamp"),
        col("timestamp_clean").alias("original_timestamp"),
        col("normalized_helpful_vote").alias("normalised_helpful_vote"),
        col("helpful_vote_clean").alias("original_helpful_vote"),
        col("verified_purchase_clean").alias("verified_purchase"),
        col("user_count").cast(IntegerType()),
        col("asin_count").cast(IntegerType()),
        col("store_category")
    )

    # Print counts for each store category
    counts = final_df.groupBy("store_category").count().collect()
    for row in counts:
        print(f"{row['store_category']}: {row['count']}")

    # Write the final DataFrame partitioned by store_category
    output_dir = "/tmp/Kindle_Store_output.parquet"
    safe_write_partitioned(final_df, output_dir, spark)
    spark.stop()

if __name__ == "__main__":
    main()
