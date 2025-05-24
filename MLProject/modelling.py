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
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set MLflow tracking URI to use local directory
mlflow.set_tracking_uri("file:./mlruns")


def load_data(file_path):
    """
    Load the preprocessed penguins dataset from CSV file
    """
    logger.info(f"Loading penguins data from {file_path}")
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Penguins data loaded successfully with shape: {data.shape}")
        logger.info(f"Columns: {list(data.columns)}")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def prepare_data(data):
    """
    Prepare features and target variable for penguins dataset
    """
    logger.info("Preparing penguins data for modeling")

    # For preprocessed penguins dataset, separate features and target
    # Assuming the dataset has 'species_encoded' as target and 'species_original' for reference
    if "species_encoded" in data.columns:
        # Use encoded species as target
        y = data["species_encoded"]
        # Drop both species columns from features
        X = data.drop(["species_encoded", "species_original"], axis=1, errors="ignore")
    elif "species" in data.columns:
        # If only 'species' column exists, encode it
        from sklearn.preprocessing import LabelEncoder

        encoder = LabelEncoder()
        y = encoder.fit_transform(data["species"])
        X = data.drop(["species"], axis=1)

        # Save encoder for later use
        with open("species_encoder.pkl", "wb") as f:
            pickle.dump(encoder, f)
        logger.info("Species encoder saved as species_encoder.pkl")
    else:
        # Fallback: assume last column is target
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(
        f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}"
    )
    logger.info(f"Number of features: {X_train.shape[1]}")
    logger.info(f"Number of classes: {len(np.unique(y))}")

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, **params):
    """
    Train a Random Forest classifier for penguins classification
    """
    logger.info("Training Random Forest model for penguins classification")

    # Default parameters optimized for penguins dataset
    default_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
    }

    # Update with any provided parameters
    default_params.update(params)

    model = RandomForestClassifier(**default_params)
    model.fit(X_train, y_train)
    logger.info("Model training completed")
    return model, default_params


def evaluate_model(model, X_test, y_test, species_names=None):
    """
    Evaluate the model and return metrics
    """
    logger.info("Evaluating penguins classification model performance")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    logger.info(f"Model accuracy: {accuracy:.4f}")

    # Get species names for better visualization
    if species_names is None:
        try:
            with open("species_encoder.pkl", "rb") as f:
                encoder = pickle.load(f)
                species_names = encoder.classes_
        except:
            species_names = [f"Species_{i}" for i in range(len(np.unique(y_test)))]

    class_report = classification_report(y_test, y_pred, target_names=species_names)
    logger.info("\nClassification Report:\n" + class_report)

    # Create and save confusion matrix visualization
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=species_names,
        yticklabels=species_names,
    )
    plt.title("Penguins Species Classification - Confusion Matrix")
    plt.ylabel("True Species")
    plt.xlabel("Predicted Species")
    plt.tight_layout()
    plt.savefig("penguins_confusion_matrix.jpg", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved confusion matrix visualization to penguins_confusion_matrix.jpg")

    # Create and save feature importance visualization
    if hasattr(model, "feature_importances_"):
        plt.figure(figsize=(12, 8))
        feature_names = (
            X_test.columns
            if hasattr(X_test, "columns")
            else [f"Feature_{i}" for i in range(X_test.shape[1])]
        )
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.title("Feature Importances - Penguins Classification")
        bars = plt.bar(
            range(X_test.shape[1]),
            importances[indices],
            align="center",
            color="skyblue",
        )
        plt.xticks(
            range(X_test.shape[1]),
            [feature_names[i] for i in indices],
            rotation=45,
            ha="right",
        )
        plt.ylabel("Importance")
        plt.xlabel("Features")

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.001,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.tight_layout()
        plt.savefig("penguins_feature_importance.jpg", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(
            "Saved feature importance visualization to penguins_feature_importance.jpg"
        )

    # Create species distribution plot
    plt.figure(figsize=(10, 6))
    unique_species, counts = np.unique(y_test, return_counts=True)
    species_labels = [
        species_names[i] if species_names else f"Species_{i}" for i in unique_species
    ]

    plt.subplot(1, 2, 1)
    plt.bar(species_labels, counts, color=["lightblue", "lightgreen", "lightcoral"])
    plt.title("True Species Distribution in Test Set")
    plt.ylabel("Count")
    plt.xticks(rotation=45)

    # Prediction accuracy by species
    plt.subplot(1, 2, 2)
    species_accuracy = []
    for i, species in enumerate(unique_species):
        mask = y_test == species
        if np.sum(mask) > 0:
            acc = accuracy_score(y_test[mask], y_pred[mask])
            species_accuracy.append(acc)
        else:
            species_accuracy.append(0)

    plt.bar(
        species_labels,
        species_accuracy,
        color=["lightblue", "lightgreen", "lightcoral"],
    )
    plt.title("Accuracy by Species")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)

    # Add accuracy values on bars
    for i, acc in enumerate(species_accuracy):
        plt.text(i, acc + 0.02, f"{acc:.3f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig("penguins_species_analysis.jpg", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved species analysis visualization to penguins_species_analysis.jpg")

    return {
        "accuracy": accuracy,
        "classification_report": class_report,
        "confusion_matrix": cm,
        "species_names": species_names,
    }


def main():
    # Set experiment name
    experiment_name = "penguins-species-classification"
    mlflow.set_experiment(experiment_name)

    # Load data
    data_file = "../penguins_preprocessed.csv"
    data = load_data(data_file)

    # Start MLflow run
    with mlflow.start_run(run_name="random_forest_penguins_v1"):
        logger.info(f"MLflow Run ID: {mlflow.active_run().info.run_id}")

        # Prepare data
        X_train, X_test, y_train, y_test = prepare_data(data)

        # Train model with penguins-specific parameters
        model_params = {
            "n_estimators": 150,
            "max_depth": 12,
            "min_samples_split": 3,
            "min_samples_leaf": 1,
            "random_state": 42,
        }
        model, used_params = train_model(X_train, y_train, **model_params)

        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)

        # Log parameters
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("dataset", "penguins")
        for param_name, param_value in used_params.items():
            mlflow.log_param(param_name, param_value)

        mlflow.log_param("train_size", X_train.shape[0])
        mlflow.log_param("test_size", X_test.shape[0])
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_classes", len(np.unique(y_train)))

        # Log metrics
        mlflow.log_metric("accuracy", metrics["accuracy"])

        # Calculate and log per-class metrics
        from sklearn.metrics import precision_recall_fscore_support

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, model.predict(X_test), average="weighted"
        )
        mlflow.log_metric("precision_weighted", precision)
        mlflow.log_metric("recall_weighted", recall)
        mlflow.log_metric("f1_weighted", f1)

        # Log model
        mlflow.sklearn.log_model(model, "penguins_classifier")

        # Log artifacts (visualizations)
        artifact_files = [
            "penguins_confusion_matrix.jpg",
            "penguins_feature_importance.jpg",
            "penguins_species_analysis.jpg",
        ]

        for artifact_file in artifact_files:
            if os.path.exists(artifact_file):
                mlflow.log_artifact(artifact_file)
                logger.info(f"Logged artifact: {artifact_file}")

        # Log model artifacts if they exist
        if os.path.exists("species_encoder.pkl"):
            mlflow.log_artifact("species_encoder.pkl")

        logger.info(
            "Penguins classification model training and evaluation complete with MLflow tracking"
        )
        logger.info(f"Final model accuracy: {metrics['accuracy']:.4f}")

        return {
            "model": model,
            "metrics": metrics,
            "run_id": mlflow.active_run().info.run_id,
        }


def predict_new_penguins(model_run_id=None, new_data=None):
    """
    Function to make predictions on new penguin data
    """
    if model_run_id:
        # Load model from MLflow
        model_uri = f"runs:/{model_run_id}/penguins_classifier"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Loaded model from MLflow run: {model_run_id}")
    else:
        logger.error("No model run ID provided")
        return None

    if new_data is not None:
        predictions = model.predict(new_data)
        probabilities = model.predict_proba(new_data)
        return predictions, probabilities
    else:
        logger.info("No new data provided for prediction")
        return None


if __name__ == "__main__":
    result = main()

    if result:
        print(f"\n{'='*50}")
        print("PENGUINS CLASSIFICATION RESULTS")
        print(f"{'='*50}")
        print(f"Model Accuracy: {result['metrics']['accuracy']:.4f}")
        print(f"MLflow Run ID: {result['run_id']}")
        print("Visualizations saved:")
        print("  - penguins_confusion_matrix.jpg")
        print("  - penguins_feature_importance.jpg")
        print("  - penguins_species_analysis.jpg")
        print(f"{'='*50}")
