"""
PySpark Structured Streaming job that:
  1. Reads raw packet records from the Kafka `raw_packets` topic
  2. Extracts flow features using a sliding window
  3. Runs the ML detection pipeline on each micro-batch
  4. Writes alerts to the `alerts` Kafka topic and Elasticsearch
"""

import json
import logging
from datetime import datetime, timezone
from typing import Iterator

import structlog
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    IntegerType,
    LongType,
    TimestampType,
    BooleanType,
)

from src.config import settings
from src.features.flow_features import FlowFeatureExtractor
from src.features.statistical_features import StatisticalFeatureExtractor
from src.features.behavioral_features import BehavioralFeatureExtractor
from src.detection.fusion_layer import FusionLayer

logger = structlog.get_logger(__name__)

RAW_PACKET_SCHEMA = StructType([
    StructField("timestamp", DoubleType()),
    StructField("src_ip", StringType()),
    StructField("dst_ip", StringType()),
    StructField("src_port", IntegerType()),
    StructField("dst_port", IntegerType()),
    StructField("protocol", StringType()),
    StructField("ip_flags", IntegerType()),
    StructField("ttl", IntegerType()),
    StructField("ip_len", IntegerType()),
    StructField("payload_len", IntegerType()),
    StructField("tcp_flags", StringType()),
    StructField("tcp_seq", LongType()),
    StructField("tcp_ack", LongType()),
    StructField("tcp_window", IntegerType()),
    StructField("icmp_type", IntegerType()),
    StructField("icmp_code", IntegerType()),
    StructField("capture_interface", StringType()),
    StructField("captured_at", StringType()),
])


class NIDSStreamProcessor:
    """
    Orchestrates the full real-time detection pipeline using PySpark
    Structured Streaming with 30-second sliding windows.
    """

    WINDOW_DURATION = "30 seconds"
    SLIDE_DURATION = "10 seconds"
    WATERMARK_DELAY = "60 seconds"

    def __init__(self) -> None:
        self.spark = self._build_spark_session()
        self.flow_extractor = FlowFeatureExtractor()
        self.stat_extractor = StatisticalFeatureExtractor()
        self.behavioral_extractor = BehavioralFeatureExtractor()
        self.fusion = FusionLayer()

    def _build_spark_session(self) -> SparkSession:
        return (
            SparkSession.builder
            .appName("NIDS-Stream-Processor")
            .config("spark.jars.packages",
                    "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
                    "org.elasticsearch:elasticsearch-spark-30_2.12:8.11.1")
            .config("spark.sql.streaming.checkpointLocation", "/tmp/nids-checkpoint")
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.streaming.backpressure.enabled", "true")
            .getOrCreate()
        )

    def _read_kafka_stream(self) -> DataFrame:
        return (
            self.spark.readStream
            .format("kafka")
            .option("kafka.bootstrap.servers", settings.KAFKA_BOOTSTRAP_SERVERS)
            .option("subscribe", settings.KAFKA_TOPIC_RAW_PACKETS)
            .option("startingOffsets", "latest")
            .option("maxOffsetsPerTrigger", 50_000)
            .option("failOnDataLoss", "false")
            .load()
        )

    def _parse_packets(self, raw_df: DataFrame) -> DataFrame:
        return (
            raw_df
            .select(
                F.from_json(
                    F.col("value").cast("string"),
                    RAW_PACKET_SCHEMA,
                ).alias("pkt")
            )
            .select("pkt.*")
            .withColumn(
                "event_time",
                F.to_timestamp(F.col("captured_at")),
            )
            .withWatermark("event_time", self.WATERMARK_DELAY)
        )

    def _aggregate_flows(self, parsed_df: DataFrame) -> DataFrame:
        return (
            parsed_df
            .groupBy(
                F.window("event_time", self.WINDOW_DURATION, self.SLIDE_DURATION),
                "src_ip",
                "dst_ip",
                "dst_port",
                "protocol",
            )
            .agg(
                F.count("*").alias("packet_count"),
                F.sum("ip_len").alias("total_bytes"),
                F.sum("payload_len").alias("total_payload_bytes"),
                F.avg("ip_len").alias("mean_packet_size"),
                F.stddev("ip_len").alias("std_packet_size"),
                F.min("timestamp").alias("flow_start"),
                F.max("timestamp").alias("flow_end"),
                F.collect_list("tcp_flags").alias("tcp_flags_list"),
                F.avg("ttl").alias("mean_ttl"),
                F.countDistinct("src_port").alias("unique_src_ports"),
                F.sum(F.when(F.col("tcp_flags").contains("S"), 1).otherwise(0)).alias("syn_count"),
                F.sum(F.when(F.col("tcp_flags").contains("R"), 1).otherwise(0)).alias("rst_count"),
                F.sum(F.when(F.col("tcp_flags").contains("F"), 1).otherwise(0)).alias("fin_count"),
            )
            .withColumn("flow_duration", F.col("flow_end") - F.col("flow_start"))
            .withColumn("bytes_per_second",
                        F.col("total_bytes") / F.greatest(F.col("flow_duration"), F.lit(1e-6)))
            .withColumn("packets_per_second",
                        F.col("packet_count") / F.greatest(F.col("flow_duration"), F.lit(1e-6)))
            .withColumn("window_start", F.col("window.start"))
            .withColumn("window_end", F.col("window.end"))
            .drop("window")
        )

    def _process_batch(self, batch_df: DataFrame, batch_id: int) -> None:
        if batch_df.isEmpty():
            return

        rows = batch_df.toPandas()
        logger.info("processing_batch", batch_id=batch_id, rows=len(rows))

        results = self.fusion.score_batch(rows)

        alerts = results[results["threat_label"] != "benign"]
        if not alerts.empty:
            self._write_alerts_to_kafka(alerts)
            self._write_flows_to_elasticsearch(rows)

    def _write_alerts_to_kafka(self, alerts_df) -> None:
        from src.ingestion.kafka_producer import NIDSProducer
        producer = NIDSProducer()
        producer.start()

        for _, row in alerts_df.iterrows():
            alert = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "src_ip": row["src_ip"],
                "dst_ip": row["dst_ip"],
                "dst_port": int(row["dst_port"]) if row["dst_port"] else None,
                "protocol": row["protocol"],
                "threat_label": row["threat_label"],
                "fusion_score": float(row["fusion_score"]),
                "ml_score": float(row.get("ml_score", 0)),
                "anomaly_score": float(row.get("anomaly_score", 0)),
                "signature_score": float(row.get("signature_score", 0)),
                "matched_rules": row.get("matched_rules", []),
            }
            producer.produce(
                topic=settings.KAFKA_TOPIC_ALERTS,
                key=f"{row['src_ip']}->{row['dst_ip']}",
                value=alert,
            )

        producer.flush()
        producer.close()

    def _write_flows_to_elasticsearch(self, flows_df) -> None:
        from src.storage.elasticsearch_client import ElasticsearchClient
        es = ElasticsearchClient()
        docs = flows_df.to_dict(orient="records")
        es.bulk_index(index=settings.ELASTICSEARCH_INDEX_FLOWS, docs=docs)

    def run(self) -> None:
        logger.info("stream_processor_starting")
        raw_df = self._read_kafka_stream()
        parsed_df = self._parse_packets(raw_df)
        aggregated_df = self._aggregate_flows(parsed_df)

        query = (
            aggregated_df.writeStream
            .outputMode("update")
            .foreachBatch(self._process_batch)
            .trigger(processingTime="10 seconds")
            .start()
        )

        logger.info("stream_processor_running")
        query.awaitTermination()


if __name__ == "__main__":
    processor = NIDSStreamProcessor()
    processor.run()
