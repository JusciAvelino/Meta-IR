# Meta-IR: Meta-Learning for Imbalanced Regression

## Introduction

Meta-Learning for Imbalanced Regression (Meta-IR) is a recommendation system designed to address imbalanced regression tasks. The goal of Meta-IR is to recommend suitable algorithms and techniques based on problem meta-features, which are used as input to the meta-classifier. The meta-classifier then suggests an optimal resampling strategy and learning algorithm, effectively creating a customized pipeline tailored to a new problem's meta-features.

# Contents
This file contains:
- **META_IR.py** with the code implemented.
- **example.py** with an example of how execute Meta-IR
- **data** with the 218 datasets.

# Steps:
**Load the Datasets:**
Begin by loading the datasets that will be used in the meta-learning process.

**Creating META_IR Instance:**
Instantiate the META_IR class by assigning it to the variable meta_ir.

```python
meta_ir = META_IR(data_sets)
```

**Install R Packages:**
Call the method meta_ir.install_rpackages() to install the necessary R packages required for the subsequent steps.

```python
meta_ir.install_rpackages()
```

**Meta-Feature Extraction:**
Extract meta-features from individual datasets using the meta_ir.meta_feature_extraction() method. The extracted features are stored in the variable m.

```python
m = meta_ir.meta_feature_extraction()
```

**Meta Target Definition:**
Define the goals of the meta-learning process using the meta_ir.meta_target_definition() method. These goals likely include information about learning algorithms and resampling strategies.

```python
meta_target = meta_ir.meta_target_definition()
```

**Data Concatenation:**
Concatenate the meta_target DataFrame with the m DataFrame using pd.concat(). This adds the 'clf' column (representing learning algorithms) and the 'strategy' column (representing resampling strategies).

```python
m = pd.concat([meta_target[['clf', 'strategy']], m], axis=1)
```

**Independent Training Approach:**
Apply the independent training approach using meta_ir.independent_training(m). This step involves using the extracted features and defined goals to train models.

```python
df_independent_training = meta_ir.independent_training(m)
```

**Model-First Approach:**
Apply the model-first approach using meta_ir.model_first(m). This approach likely focuses on training models before considering specific strategies.

```python
df_model_first = meta_ir.model_first(m)
```

**Strategy-First Approach:**
Apply the strategy-first approach using meta_ir.strategy_first(m). This approach likely prioritizes strategies over specific models.

```python
df_strategy_first = meta_ir.strategy_first(m)
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

