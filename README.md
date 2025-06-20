# Penguins Classification with MLflow and GitHub Actions CI

This project demonstrates how to create a machine learning workflow for the Penguins dataset using MLflow and GitHub Actions for continuous integration (CI).

## Project Structure

```
Workflow-CI/
│
├── .github/workflows/
│   └── train_model.yml              # GitHub Actions workflow configuration
│
├── MLProject/                       # MLflow project folder
│   ├── MLProject                    # MLflow project definition file
│   ├── modelling.py                 # Main model training script
│   ├── conda.yaml                   # Conda environment specification
│   └── penguins_preprocessed.csv    # Preprocessed dataset
│
└── README.md                        # Project documentation
```

## How It Works

1. **MLflow Project**: The `MLProject` folder contains an MLflow project that defines how to train the Penguins classification model. The project includes:
   - A main script (`modelling.py`) for training and evaluating the model
   - Environment specification (`conda.yaml`)
   - Project definition (`MLProject` file)

2. **GitHub Actions Workflow**: The workflow is defined in `.github/workflows/train_model.yml` and is triggered:
   - On push to the main branch
   - On pull requests to the main branch
   - Manually via workflow dispatch

3. **Model Training Process**: When the workflow runs, it:
   - Sets up a Python environment
   - Installs required dependencies
   - Runs the MLflow project to train the model
   - Uploads artifacts (model, metrics, visualizations) for review

## How to Use

### Local Development

1. Clone this repository:
   ```
   git clone https://github.com/agungknwn/ML_workflow-CI.git
   cd ML_Workflow-CI
   ```

2. Run the MLflow project locally:
   ```
   cd MLProject
   conda env create -f conda.yaml
   conda activate penguins-classification
   mlflow run . --experiment-name penguins-species-classification
   ```

### Trigger CI Workflow

1. Push changes to the main branch:
   ```
   git add .
   git commit -m "Update model or data"
   git push origin main
   ```

2. Or manually trigger the workflow from the GitHub Actions tab in your repository.

### View Results

1. Check the GitHub Actions tab in your repository to see workflow runs
2. Download artifacts from completed workflow runs to view model results

## Requirements

- Python 3.9+
- MLflow
- scikit-learn
- pandas
- matplotlib
- seaborn
- GitHub account with Actions enabled
