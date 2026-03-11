#!/usr/bin/env python3
import json
import io
import boto3
import pandas as pd
from datetime import datetime

from baseline import BaselineManager
from detector import AnomalyDetector
import logging
logger = logging.getLogger(__name__)
s3 = boto3.client("s3")

NUMERIC_COLS = ["temperature", "humidity", "pressure", "wind_speed"]  # students configure this

def process_file(bucket: str, key: str):
    print(f"Processing: s3://{bucket}/{key}")

    # 1. Download raw file
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(io.BytesIO(response["Body"].read()))
        logger.info(f"Loaded {len(df)} rows, columns: {list(df.columns)}")
        print(f"  Loaded {len(df)} rows, columns: {list(df.columns)}")
    except Exception as e:
        logger.error(f"Failed to download or parse s3://{bucket}/{key}: {e}")
        print(f"ERROR downloading {key}: {e}")
        return None

    # 2. Load current baseline
    try:
        baseline_mgr = BaselineManager(bucket=bucket)
        baseline = baseline_mgr.load()
        logger.info("Baseline loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load baseline: {e}")
        print(f"ERROR loading baseline: {e}")
        return None

    # 3. Update baseline with values from this batch BEFORE scoring
    #    (use only non-null values for each channel)
    for col in NUMERIC_COLS:
        if col in df.columns:
            try:
                clean_values = df[col].dropna().tolist()
                if clean_values:
                    baseline = baseline_mgr.update(baseline, col, clean_values)
                    logger.info(f"Baseline updated for channel '{col}' with {len(clean_values)} values.")
            except Exception as e:
                logger.error(f"Failed to update baseline for column '{col}': {e}")
                print(f"ERROR updating baseline for {col}: {e}")

    # 4. Run detection
    try:
        detector = AnomalyDetector(z_threshold=3.0, contamination=0.05)
        scored_df = detector.run(df, NUMERIC_COLS, baseline, method="both")
        anomaly_count = int(scored_df["anomaly"].sum()) if "anomaly" in scored_df.columns else 0
        logger.info(f"Detection complete: {anomaly_count}/{len(df)} anomalies flagged.")
        print(f"  Done: {anomaly_count}/{len(df)} anomalies flagged")
    except Exception as e:
        logger.error(f"Anomaly detection failed on {key}: {e}")
        print(f"ERROR during detection on {key}: {e}")
        return None

    # 5. Write scored file to processed/ prefix
    output_key = key.replace("raw/", "processed/")
    try:
        csv_buffer = io.StringIO()
        scored_df.to_csv(csv_buffer, index=False)
        s3.put_object(
            Bucket=bucket,
            Key=output_key,
            Body=csv_buffer.getvalue(),
            ContentType="text/csv"
        )
        logger.info(f"Scored file written to s3://{bucket}/{output_key}")
    except Exception as e:
        logger.error(f"Failed to write scored file to s3://{bucket}/{output_key}: {e}")
        print(f"ERROR writing scored file {output_key}: {e}")

    # 6. Save updated baseline back to S3
    try:
        baseline_mgr.save(baseline)
        logger.info("Baseline saved to S3.")
    except Exception as e:
        logger.error(f"Failed to save baseline to S3: {e}")
        print(f"ERROR saving baseline: {e}")

    # 7. Build and return a processing summary
    try:
        summary = {
            "source_key": key,
            "output_key": output_key,
            "processed_at": datetime.utcnow().isoformat(),
            "total_rows": len(df),
            "anomaly_count": anomaly_count,
            "anomaly_rate": round(anomaly_count / len(df), 4) if len(df) > 0 else 0,
            "baseline_observation_counts": {
                col: baseline.get(col, {}).get("count", 0) for col in NUMERIC_COLS
            }
        }
        summary_key = output_key.replace(".csv", "_summary.json")
        s3.put_object(
            Bucket=bucket,
            Key=summary_key,
            Body=json.dumps(summary, indent=2),
            ContentType="application/json"
        )
        logger.info(f"Summary written to s3://{bucket}/{summary_key}")
    except Exception as e:
        logger.error(f"Failed to write summary JSON: {e}")
        print(f"ERROR writing summary JSON: {e}")
        return None

    print(f"  Done: {anomaly_count}/{len(df)} anomalies flagged")
    return summary
