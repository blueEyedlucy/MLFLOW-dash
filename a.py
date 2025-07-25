


import os
import time
import uuid
import logging

import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import numpy as np

# ── CONFIG & AUTH ────────────────────────────────────────────────────────────

# 1. Credentials (env‐var basic auth for MLflow Server)
os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "mlflow_Str0ng_Admin@2025"

# 2. Point at your remote MLflow server & experiment
mlflow.set_tracking_uri("https://mlflow.dev.cygeniq.com/")
mlflow.set_experiment("GRC")

# 3. Turn on scikit‐learn autologging
mlflow.sklearn.autolog()

# 4. Create a reusable client for registry operations
client = MlflowClient()

# 5. Setup basic logging for local trace output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── HYPERPARAMETERS ───────────────────────────────────────────────────────────

params = {
    "learning_rate": 0.1,
    "max_depth": 5,
    "num_trees": 100,
    "regularization_strength": 0.01,
    "temperature": 0.5,
    "noise_variance": 0.1,
    "token_window_size": 512,
}

# ── DATA PREP ─────────────────────────────────────────────────────────────────

X, y_true = make_blobs(
    n_samples=100, centers=3, n_features=5, random_state=42
)

# ── TRAIN + LOGGING RUN ────────────────────────────────────────────────────────

with mlflow.start_run(run_name="GRC_TEST") as run:
    run_id = run.info.run_id
    trace_id = str(uuid.uuid4())
    mlflow.set_tag("trace_id", trace_id)
    logger.info(f"Started MLflow run {run_id} with trace_id {trace_id}")

    # 1) Log extra hyperparameters
    mlflow.log_params(params)
    logger.info(f"Logged hyperparameters: {params}")

    # 2) Time the training for latency metric
    t0 = time.time()
    model = KMeans(n_clusters=3, random_state=42).fit(X)
    latency = time.time() - t0
    mlflow.log_metric("latency", latency)
    logger.info(f"Training latency: {latency:.4f}s")

    # 3) Core clustering metric
    sil_score = silhouette_score(X, model.labels_)
    mlflow.log_metric("silhouette_score", sil_score)
    logger.info(f"Silhouette score: {sil_score:.4f}")

    # 4) Treat cluster IDs vs true labels as a “classification” to get accuracy/precision/etc.
    #    Note: you may need to align cluster IDs → true labels in real scenarios.
    accuracy = accuracy_score(y_true, model.labels_)
    precision = precision_score(y_true, model.labels_, average="macro")
    recall = recall_score(y_true, model.labels_, average="macro")
    f1 = f1_score(y_true, model.labels_, average="macro")
    # Simple “bias score” as the imbalance of cluster sizes
    counts = np.bincount(model.labels_)
    bias_score = float(np.std(counts / counts.sum()))
    mlflow.log_metrics({
        "accuracy":       accuracy,
        "precision":      precision,
        "recall":         recall,
        "f1_score":       f1,
        "bias_score":     bias_score,
    })
    logger.info(f"Logged classification metrics: "
                f"acc={accuracy:.4f}, prec={precision:.4f}, "
                f"rec={recall:.4f}, f1={f1:.4f}, bias={bias_score:.4f}")

    # 5) Register the model artifact
    model_uri = f"runs:/{run_id}/model"
    model_name = "GRC_MODEL"
    try:
        mv = mlflow.register_model(model_uri, model_name)
        version = mv.version
        logger.info(f"Registered model {model_name} at version {version}")
    except MlflowException as e:
        logger.error(f"Model registration failed: {e}")
        raise

    # 6) Transition newly registered version to “Staging”
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Staging",
        archive_existing_versions=True,
    )
    logger.info(f"Transitioned model {model_name} v{version} to Staging")

print("Done ✅")
