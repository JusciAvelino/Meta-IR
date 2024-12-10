# Meta-IR: Meta-Learning for Imbalanced Regression

## Introduction

Meta-Learning for Imbalanced Regression (Meta-IR) is a recommendation system designed to address imbalanced regression tasks. The goal of Meta-IR is to recommend suitable algorithms and techniques based on problem meta-features, which are used as input to the meta-classifier. The meta-classifier then suggests an optimal resampling strategy and learning algorithm, effectively creating a customized pipeline tailored to a new problem's meta-features.

## Contents
This file contains:
- **META_IR.py** with the code implemented.
- **data** with the 218 datasets.

## Example:

```python
def run_full_meta_ir_pipeline(folder_path):
    """
    Executes the full META_IR pipeline, including concatenation of meta-targets
    and applying independent training, model-first, and strategy-first evaluations.

    Args:
        folder_path (str): Path to the folder containing the datasets.
    """
    # Load datasets from the folder
    print(f"Loading datasets from folder: {folder_path}")
    data_sets = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]
    if not data_sets:
        raise ValueError("No CSV files found in the specified folder.")

    # Instantiate the META_IR class
    meta_ir = META_IR(data_sets=data_sets)

    #Installs the required R packages for the META_IR pipeline.
    print("Installing R packages...")
    meta_ir.install_rpackages()
    print("R packages installed successfully.")

    # Extract meta-features
    print("Extracting meta-features...")
    meta_features = meta_ir.meta_feature_extraction()

    # Define meta-targets
    print("Defining meta-targets...")
    meta_target = meta_ir.meta_target_definition()

    sera_meta_target = meta_target[meta_target['metric'] == 'sera']
    f1_meta_target = meta_target[meta_target['metric'] == 'f1score']

    print("Concatenating meta-targets with meta-features...")

    m_sera = pd.concat(
    [sera_meta_target[['dataset', 'model', 'strategy']].reset_index(drop=True),
     meta_features.reset_index(drop=True)],
    axis=1
    )

    m_f1 = pd.concat(
    [f1_meta_target[['dataset', 'model', 'strategy']].reset_index(drop=True),
     meta_features.reset_index(drop=True)],
    axis=1
    )

    # Run independent training
    print("Running independent training...")
    df_independent_training = meta_ir.independent_training(m_sera)
    print("Independent training results:")
    print(df_independent_training)

    # Run model-first evaluation
    print("Running model-first evaluation...")
    df_model_first = meta_ir.model_first(m_sera)
    print("Model-first evaluation results:")
    print(df_model_first)

    # Run strategy-first evaluation
    print("Running strategy-first evaluation...")
    df_strategy_first = meta_ir.strategy_first(m_sera)
    print("Strategy-first evaluation results:")
    print(df_strategy_first)

    print("META_IR pipeline execution completed.")

# Path to the folder containing datasets
folder_path = "/content/drive/MyDrive/Colab Notebooks/ds_30"  # Replace with your folder path

# Execute the full pipeline
run_full_meta_ir_pipeline(folder_path)
```

## Dependencies:

To run this project you will need the following dependencies:

* Python >= 3.6
* pandas
* numpy
* scikit-learn
* rpy2
* smogn
* resreg
* ImbalancedLearningRegression
* UBL (R Package)
* IRon (R Package)
* uba (R Package)

