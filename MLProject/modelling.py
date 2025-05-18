import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set MLflow tracking URI to use the local directory
mlflow.set_tracking_uri("file:./mlruns")


def load_data(file_path):
    """
    Load the preprocessed dataset from CSV file
    """
    logger.info(f"Loading data from {file_path}")
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully with shape: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def prepare_data(data):
    """
    Prepare features and target variable
    """
    logger.info("Preparing data for modeling")

    # For Iris dataset, typically the last column is the target variable
    X = data.iloc[:, :-1]  # All columns except the last one as features
    y = data.iloc[:, -1]  # Last column as target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    logger.info(
        f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}"
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Train a Random Forest classifier
    """
    logger.info("Training Random Forest model")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    logger.info("Model training completed")
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and return metrics
    """
    logger.info("Evaluating model performance")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    logger.info(f"Model accuracy: {accuracy:.4f}")
    logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred))

    # Create and save confusion matrix visualization
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig("screenshot_artefak.jpg")
    logger.info("Saved confusion matrix visualization to screenshot_artefak.jpg")

    # Create and save feature importance visualization
    if hasattr(model, "feature_importances_"):
        plt.figure(figsize=(10, 6))
        feature_names = X_test.columns
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.title("Feature Importances")
        plt.bar(range(X_test.shape[1]), importances[indices], align="center")
        plt.xticks(
            range(X_test.shape[1]), [feature_names[i] for i in indices], rotation=90
        )
        plt.tight_layout()
        plt.savefig("screenshot_dashboard.jpg")
        logger.info(
            "Saved feature importance visualization to screenshot_dashboard.jpg"
        )

    return {
        "accuracy": accuracy,
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": cm,
    }


def main():
    # Set experiment name
    experiment_name = "iris-classification"
    mlflow.set_experiment(experiment_name)

    # Load data - adjust path to look for data in the correct location
    data_file = "../irisdataset_preprocessing.csv"
    # If file not found in parent directory, try current directory
    if not os.path.exists(data_file):
        data_file = "irisdataset_preprocessing.csv"

    data = load_data(data_file)

    # Enable MLflow autologging
    mlflow.sklearn.autolog()

    # Start MLflow run
    with mlflow.start_run(run_name="random_forest_basic"):
        logger.info(f"MLflow Run ID: {mlflow.active_run().info.run_id}")

        # Prepare data
        X_train, X_test, y_train, y_test = prepare_data(data)

        # Train model
        model = train_model(X_train, y_train)

        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)

        # Log parameters
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)

        # Log metrics
        mlflow.log_metric("accuracy", metrics["accuracy"])

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Log artifacts
        mlflow.log_artifact("screenshot_dashboard.jpg")
        mlflow.log_artifact("screenshot_artefak.jpg")

        logger.info("Model training and evaluation complete with MLflow tracking")


if __name__ == "__main__":
    main()
