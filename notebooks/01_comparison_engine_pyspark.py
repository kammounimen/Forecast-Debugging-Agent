# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 01 - Moteur de comparaison (PySpark)
# MAGIC Même logique que la version Scala (`01_comparison_engine.scala`), compatible **Free Edition** (Scala non supporté).
# MAGIC Écrit 4 tables Gold : `cmp_forecast_gap`, `cmp_inputs`, `cmp_freshness`, `cmp_case_summary`.
# MAGIC Seuils (documentés dans le RAG) : tolérance d'écart 5 %, fraîcheur 24 h.

# COMMAND ----------

from pyspark.sql import functions as F

catalog = spark.sql("SELECT current_catalog()").first()[0]
schema = "forecast_debug"
base = f"{catalog}.{schema}"

TOLERANCE_PCT = 5.0
STALE_HOURS = 24.0
ACCEPTABLE_HOURS = 6.0

def save(df, name):
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{base}.{name}")
    print(f"Créée : {base}.{name} ({df.count()} lignes)")

# COMMAND ----------

# 1. Écart de prévision
fc = spark.table(f"{base}.forecast_predictions").withColumn("predicted_units", F.col("predicted_units").cast("double"))

forecast_gap = (
    fc.groupBy("case_id", "product_id")
    .agg(
        F.max(F.when(F.col("system") == "BATCH", F.col("predicted_units"))).alias("batch_forecast"),
        F.max(F.when(F.col("system") == "REALTIME", F.col("predicted_units"))).alias("realtime_forecast"),
    )
    .withColumn("difference", F.col("realtime_forecast") - F.col("batch_forecast"))
    .withColumn("difference_pct", F.round(100.0 * F.col("difference") / F.col("batch_forecast"), 2))
    .withColumn("within_tolerance", F.abs(F.col("difference_pct")) <= TOLERANCE_PCT)
)
save(forecast_gap, "cmp_forecast_gap")

# COMMAND ----------

# 2. Différences de features et de configurations
def diff_table(table, name_col, value_col, finding_type):
    df = spark.table(f"{base}.{table}")
    def side(system, alias):
        return (df.filter(F.col("system") == system)
                  .select("case_id", F.col(name_col).alias("name"), F.col(value_col).alias(alias)))
    return (
        side("BATCH", "batch_value")
        .join(side("REALTIME", "realtime_value"), ["case_id", "name"], "full_outer")
        .withColumn("finding_type", F.lit(finding_type))
        .withColumn("is_difference",
                    F.coalesce(F.col("batch_value") != F.col("realtime_value"), F.lit(True)))
        .select("case_id", "finding_type", "name", "batch_value", "realtime_value", "is_difference")
    )

inputs = diff_table("feature_snapshots", "feature_name", "feature_value", "FEATURE").unionByName(
    diff_table("model_configs", "config_name", "config_value", "CONFIG"))
save(inputs, "cmp_inputs")

# COMMAND ----------

# 3. Fraîcheur : âge de chaque snapshot au moment de l'exécution de son pipeline
runs = (
    spark.table(f"{base}.pipeline_runs")
    .withColumn("snapshot_ts", F.to_timestamp("data_snapshot_timestamp"))
    .withColumn("run_ts", F.to_timestamp("run_timestamp"))
    .withColumn("snapshot_age_hours", (F.unix_timestamp("run_ts") - F.unix_timestamp("snapshot_ts")) / 3600.0)
)

def run_side(system, p):
    return runs.filter(F.col("system") == system).select(
        "case_id",
        F.col("snapshot_ts").alias(f"{p}_snapshot"),
        F.col("snapshot_age_hours").alias(f"{p}_age_hours"),
        F.col("status").alias(f"{p}_status"),
        F.col("warning_message").alias(f"{p}_warning"),
        F.col("data_version").alias(f"{p}_data_version"),
    )

freshness = (
    run_side("BATCH", "batch").join(run_side("REALTIME", "realtime"), "case_id")
    .withColumn("snapshot_gap_hours",
                F.round(F.abs(F.unix_timestamp("batch_snapshot") - F.unix_timestamp("realtime_snapshot")) / 3600.0, 2))
    .withColumn("max_age_hours", F.greatest("batch_age_hours", "realtime_age_hours"))
    .withColumn("freshness_status",
                F.when(F.col("max_age_hours") > STALE_HOURS, "STALE")
                 .when(F.col("max_age_hours") > ACCEPTABLE_HOURS, "ACCEPTABLE")
                 .otherwise("FRESH"))
)
save(freshness, "cmp_freshness")

# COMMAND ----------

# 4. Synthèse par cas
input_counts = inputs.groupBy("case_id").agg(
    F.sum(F.when((F.col("finding_type") == "FEATURE") & F.col("is_difference"), 1).otherwise(0)).alias("n_feature_diffs"),
    F.sum(F.when((F.col("finding_type") == "CONFIG") & F.col("is_difference"), 1).otherwise(0)).alias("n_config_diffs"),
)

summary = (
    forecast_gap.join(input_counts, "case_id")
    .join(freshness.select("case_id", "snapshot_gap_hours", "freshness_status"), "case_id")
    .orderBy("case_id")
)
save(summary, "cmp_case_summary")
display(summary)