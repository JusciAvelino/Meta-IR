# Meta-IR: Meta-Learning for Imbalanced Regression

## Introduction

Meta-Learning for Imbalanced Regression (Meta-IR) is a recommendation system designed to address imbalanced regression tasks. The goal of Meta-IR is to recommend suitable algorithms and techniques based on problem meta-features, which are used as input to the meta-classifier. The meta-classifier then suggests an optimal resampling strategy and learning algorithm, effectively creating a customized pipeline tailored to a new problem's meta-features.

# Contents
This file contains:
- **META_IR.py** with the code implemented.
- **example.py** with an example of how execute Meta-IR
- **data** with the 218 datasets.

## How Meta-IR Works

Meta-IR operates by analyzing the characteristics of a given dataset to determine the appropriate resampling strategy and learning algorithm. We propose two distinct formulations:

1. **Independent Formulation:** In this approach, the meta-classifiers are trained separately to determine the best learning algorithm and the optimal resampling strategy for a given dataset.

2. **Chained Formulation:** The chained approach involves a sequential procedure where the output of one meta-classifier serves as input for another.

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

