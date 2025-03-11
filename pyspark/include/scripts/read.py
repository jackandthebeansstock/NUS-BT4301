def main():
    import os
    #if local THESE SHOULD HAVE BEEN SET
    # set "PYSPARK_PYTHON=C:\Program Files\Python310\python.exe"
    # set "PYSPARK_DRIVER_PYTHON=C:\Program Files\Python310\python.exe"
    # set "JAVA_HOME=C:\Program Files\Java\jdk-17"
    # set "SPARK_HOME=C:\hadoop\spark-3.5.5-bin-hadoop3\spark-3.5.5-bin-hadoop3"
    # set "HADOOP_HOME=C:\hadoop"
    # set "WINUTILS=C:\hadoop\bin\winutils.exe"
    #if local these should be manually declared
    # os.environ["PYSPARK_PYTHON"] = r"C:\Program Files\Python310\python.exe"
    # os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Program Files\Python310\python.exe"
    # os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk-17"
    # os.environ["SPARK_HOME"] = r"C:\hadoop\spark-3.5.5-bin-hadoop3\spark-3.5.5-bin-hadoop3"
    # os.environ["HADOOP_HOME"] = r"C:\hadoop"
    # os.environ["WINUTILS"] = r"C:\hadoop\bin\winutils.exe"

    import pandas as pd #for retrieving the file due to errors in running parquet in spark (due to winutils)
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, when, round as spark_round, trim, lit, expr, min as spark_min, max as spark_max
    from pyspark.sql.window import Window

    # Create Spark session assuming there is no well-configured Spark cluster or entrypoint that manages sessions
    spark = SparkSession.builder \
        .appName("KindleStorePreprocessing") \
        .config("spark.ui.port", "7077") \
        .config("spark.driver.memory", "4g") \
        .config("spark.rpc.message.maxSize", "512") \
        .getOrCreate()

        # .config("spark.driver.memory", "4g") \

    # Input path
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(BASE_DIR, "..", "Kindle_Store.parquet")
    # input_path = r"./include/Kindle_Store.parquet"
    df = spark.read.parquet(input_path)

    #preprocessing 
    #items to drop
    df = df.drop("images")  
    #data cleaning + typecasting
    df = df.withColumn("rating_clean",
        when(col("rating") < 0, lit(0))
        .when(col("rating") > 5, lit(5))
        .otherwise(spark_round(col("rating")))
        .cast("integer")
    )
    df = df.withColumn("title_clean",
        when((col("title").isNull()) | (trim(col("title")) == ""), lit(""))
        .otherwise(col("title").cast("string"))
    )
    df = df.withColumn("text_clean",
        when((col("text").isNull()) | (trim(col("text")) == ""), lit(""))
        .otherwise(col("text").cast("string"))
    )
    df = df.withColumn("asin_clean", col("asin").cast("string"))
    df = df.withColumn("parent_asin_clean", col("parent_asin").cast("string"))
    df = df.withColumn("user_id_clean", col("user_id").cast("string"))
    df = df.withColumn("verified_purchase_clean", col("verified_purchase").cast("boolean"))
    df = df.withColumn("timestamp_clean", col("timestamp").cast("long"))
    df = df.withColumn("timestamp_days", (col("timestamp") / 86400000).cast("float"))
    df = df.withColumn("helpful_vote_clean", col("helpful_vote").cast("integer"))


    # ----- Normalization -----
    # We now want to normalize 'timestamp_days' and 'helpful_vote_clean' to range [0,1].

    # Compute global min and max values for these columns
    agg_vals = df.agg(
        spark_min("timestamp_days").alias("min_ts"),
        spark_max("timestamp_days").alias("max_ts"),
        spark_min("helpful_vote_clean").alias("min_hv"),
        spark_max("helpful_vote_clean").alias("max_hv")
    ).collect()[0]

    min_ts = agg_vals["min_ts"]
    max_ts = agg_vals["max_ts"]
    min_hv = agg_vals["min_hv"]
    max_hv = agg_vals["max_hv"]

    # Create normalized columns. Use a when clause to avoid division by zero.
    df = df.withColumn("normalized_timestamp",
        when(lit(max_ts) - lit(min_ts) == 0, lit(0.0))
        .otherwise((col("timestamp_days") - lit(min_ts)) / (lit(max_ts) - lit(min_ts)))
    )

    df = df.withColumn("normalized_helpful_vote",
        when(lit(max_hv) - lit(min_hv) == 0, lit(0.0))
        .otherwise((col("helpful_vote_clean") - lit(min_hv)) / (lit(max_hv) - lit(min_hv)))
    )

    #count user id and parent asin id
    user_window = Window.partitionBy("user_id_clean")
    df = df.withColumn("count_user_id", expr("count(*) over (partition by user_id_clean)"))
    asin_window = Window.partitionBy("parent_asin_clean")
    df = df.withColumn("count_parent_asin", expr("count(*) over (partition by parent_asin_clean)"))


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
        col("count_user_id").cast("integer"),
        col("count_parent_asin").cast("integer")
    )

    # Dataset 1: Both counts greater than 2
    df1 = final_df.filter((col("count_user_id") > 2) & (col("count_parent_asin") > 2))
    # Dataset 2: Both counts exactly equal to 2
    df2 = final_df.filter((col("count_user_id") == 2) & (col("count_parent_asin") == 2))
    # Dataset 3: Either count less than 2
    df3 = final_df.filter((col("count_user_id") < 2) | (col("count_parent_asin") < 2))

    # output_dir1 = r"./include/Kindle_Store_valid.parquet"
    # output_dir2 = r"./include/Kindle_Store_partiallyvalid.parquet"
    # output_dir3 = r"./include/Kindle_Store_invalid.parquet"
    
    output_dir1 = os.path.join(BASE_DIR, "..", "Kindle_Store_valid.parquet")
    output_dir2 = os.path.join(BASE_DIR, "..", "Kindle_Store_partiallyvalid.parquet")
    output_dir3 = os.path.join(BASE_DIR, "..", "Kindle_Store_invalid.parquet")
    
    df1.show()

    overwrite_parquet(df1, output_dir1)
    del df1
    overwrite_parquet(df2, output_dir2)
    del df2
    overwrite_parquet(df3, output_dir3)
    del df3

    spark.stop()
    return None

def overwrite_parquet(df, output_file):
    #spark native solution does not work
    #df.coalesce(1).write.format("parquet").mode("overwrite").save(output_file)
    df.coalesce(1).write.mode("overwrite").parquet(output_file)
    # df = df.coalesce(1).toPandas()
    # import os
    # import pandas as pd
    # if os.path.exists(output_file):
    #     df = pd.concat([pd.read_parquet(output_file), df], ignore_index=True)
    # df.to_parquet(output_file, index=False)
    return None

if __name__ == "__main__":
    main()