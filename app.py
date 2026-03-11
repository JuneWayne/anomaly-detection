# app.py
import io
import json
import os
import boto3
import pandas as pd
import requests
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, Request
from baseline import BaselineManager
from processor import process_file
import logging

# setting up logging
log_file = "app.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ],
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Anomaly Detection Pipeline")

s3 = boto3.client("s3")
BUCKET_NAME = os.environ["BUCKET_NAME"]
# log the bucket name at startup for easier debugging and verification in logs
logger.info(f"Application started with bucket: {BUCKET_NAME}")

# ── SNS subscription confirmation + message handler ──────────────────────────

@app.post("/notify")
async def handle_sns(request: Request, background_tasks: BackgroundTasks):
    # This endpoint handles both SNS subscription confirmations and notifications about new S3 files.
    try:
        body = await request.json()
        msg_type = request.headers.get("x-amz-sns-message-type")

        if msg_type == "SubscriptionConfirmation":
            confirm_url = body["SubscribeURL"]
            try:
                requests.get(confirm_url, timeout=10)
                # log the successful confirmation of the SNS subscription
                logger.info("SNS subscription confirmed successfully.")
            except Exception as e:
                # log the error if the SNS subscription confirmation fails
                logger.error(f"Failed to confirm SNS subscription: {e}")
                print(f"ERROR confirming SNS subscription: {e}")
            return {"status": "confirmed"}

        if msg_type == "Notification":
            try:
                s3_event = json.loads(body["Message"])
                for record in s3_event.get("Records", []):
                    key = record["s3"]["object"]["key"]
                    if key.startswith("raw/") and key.endswith(".csv"):
                        # log the detection of a new file and the queuing of its processing
                        logger.info(f"New file arrived: {key} — queuing for processing.")
                        background_tasks.add_task(process_file, BUCKET_NAME, key)
            except Exception as e:
                # log the error if parsing the SNS notification fails, which could indicate issues with the message format or content
                logger.error(f"Failed to parse SNS notification body: {e}")
                print(f"ERROR parsing SNS notification: {e}")

    except Exception as e:
        # log any unexpected errors that occur while handling the /notify request, which could be due to issues with the request itself or other unforeseen problems
        logger.error(f"Unexpected error in /notify: {e}")
        print(f"ERROR in /notify handler: {e}")

    return {"status": "ok"}


# ── Query endpoints ───────────────────────────────────────────────────────────

@app.get("/anomalies/recent")
def get_recent_anomalies(limit: int = 50):
    """Return rows flagged as anomalies across the 10 most recent processed files."""
    # log the start of the retrieval of recent anomalies, including the limit parameter for better traceability in logs
    try:
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix="processed/")

        keys = sorted(
            [
                obj["Key"]
                for page in pages
                for obj in page.get("Contents", [])
                if obj["Key"].endswith(".csv")
            ],
            reverse=True,
        )[:10]

        all_anomalies = []
        for key in keys:
            try:
                response = s3.get_object(Bucket=BUCKET_NAME, Key=key)
                df = pd.read_csv(io.BytesIO(response["Body"].read()))
                if "anomaly" in df.columns:
                    flagged = df[df["anomaly"] == True].copy()
                    flagged["source_file"] = key
                    all_anomalies.append(flagged)
            except Exception as e:
                # log the error if reading a processed file fails, which could indicate issues with the file's existence, permissions, or format
                logger.error(f"Failed to read processed file {key}: {e}")
                print(f"ERROR reading {key}: {e}")

        if not all_anomalies:
            return {"count": 0, "anomalies": []}

        combined = pd.concat(all_anomalies).head(limit)
        return {"count": len(combined), "anomalies": combined.to_dict(orient="records")}

    except Exception as e:
        # log any unexpected errors that occur while retrieving recent anomalies, which could be due to issues with S3 access, data processing, or other unforeseen problems
        logger.error(f"Error in /anomalies/recent: {e}")
        print(f"ERROR in /anomalies/recent: {e}")
        return {"error": str(e)}

@app.get("/anomalies/summary")
def get_anomaly_summary():
    """Aggregate anomaly rates across all processed files using their summary JSONs."""
    try:
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix="processed/")

        summaries = []
        for page in pages:
            for obj in page.get("Contents", []):
                if obj["Key"].endswith("_summary.json"):
                    try:
                        response = s3.get_object(Bucket=BUCKET_NAME, Key=obj["Key"])
                        summaries.append(json.loads(response["Body"].read()))
                    except Exception as e:
                        # log the error if reading a summary file fails, which could indicate issues with the file's existence, permissions, or format, and is crucial for understanding potential gaps in the anomaly summary data
                        logger.error(f"Failed to read summary file {obj['Key']}: {e}")
                        print(f"ERROR reading summary {obj['Key']}: {e}")

        if not summaries:
            return {"message": "No processed files yet."}

        total_rows = sum(s["total_rows"] for s in summaries)
        total_anomalies = sum(s["anomaly_count"] for s in summaries)

        return {
            "files_processed": len(summaries),
            "total_rows_scored": total_rows,
            "total_anomalies": total_anomalies,
            "overall_anomaly_rate": round(total_anomalies / total_rows, 4) if total_rows > 0 else 0,
            "most_recent": sorted(
            summaries, key=lambda x: x["processed_at"], reverse=True)[:5],
        }

    except Exception as e:
        # log any unexpected errors that occur while retrieving the anomaly summary, which could be due to issues with S3 access, data processing, or other unforeseen problems, and is important for diagnosing issues with the summary endpoint
        logger.error(f"Error in /anomalies/summary: {e}")
        print(f"ERROR in /anomalies/summary: {e}")
        return {"error": str(e)}


@app.get("/baseline/current")
def get_current_baseline():
    """Show the current per-channel statistics the detector is working from."""
    # log the start of the retrieval of the current baseline, which is important for understanding when and how often this endpoint is being accessed in the logs
    try:
        baseline_mgr = BaselineManager(bucket=BUCKET_NAME)
        baseline = baseline_mgr.load()

        channels = {}
        for channel, stats in baseline.items():
            if channel == "last_updated":
                continue
            channels[channel] = {
                "observations": stats["count"],
                "mean": round(stats["mean"], 4),
                "std": round(stats.get("std", 0.0), 4),
                "baseline_mature": stats["count"] >= 30,
            }

        return {
            "last_updated": baseline.get("last_updated"),
            "channels": channels,
        }

    except Exception as e:
        # log any unexpected errors that occur while retrieving the current baseline, which could be due to issues with S3 access, data processing, or other unforeseen problems, and is crucial for diagnosing issues with the baseline retrieval process
        logger.error(f"Error in /baseline/current: {e}")
        print(f"ERROR in /baseline/current: {e}")
        return {"error": str(e)}


@app.get("/health")
def health():
    return {"status": "ok", "bucket": BUCKET_NAME, "timestamp": datetime.utcnow().isoformat()}
