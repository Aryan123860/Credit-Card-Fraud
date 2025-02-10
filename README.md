# Credit Card Fraud Detection using XGBoost with Simulated Annealing

This repository contains a Jupyter notebook that demonstrates how to use the XGBoost algorithm for credit card fraud detection. The notebook employs a heuristic search technique called **Simulated Annealing** to efficiently find a good combination of hyper-parameters for the XGBoost model. The dataset used is from Kaggle and contains credit card transactions labeled as fraudulent or non-fraudulent.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Data Exploration](#data-exploration)
4. [Data Partitioning](#data-partitioning)
5. [XGBoost Model Setup](#xgboost-model-setup)
6. [Simulated Annealing for Hyper-parameter Tuning](#simulated-annealing-for-hyper-parameter-tuning)
7. [Results](#results)
8. [Conclusion](#conclusion)

## Introduction

The goal of this project is to detect fraudulent credit card transactions using the XGBoost algorithm. XGBoost is a powerful machine learning algorithm that is particularly effective for structured data. However, it has many hyper-parameters that need to be tuned for optimal performance. Instead of using a grid search or random search, this notebook uses **Simulated Annealing**, a heuristic search method, to efficiently explore the hyper-parameter space.

## Dataset

The dataset used in this notebook is the [Credit Card Fraud Detection dataset](https://www.kaggle.com/dalpozz/creditcardfraud) from Kaggle. It contains credit card transactions made by European cardholders in September 2013. The dataset is highly imbalanced, with only 492 fraudulent transactions out of 284,807 total transactions.

### Features:
- **Time**: Number of seconds elapsed between this transaction and the first transaction in the dataset.
- **Amount**: Transaction amount.
- **V1-V28**: Features obtained through Principal Component Analysis (PCA) for confidentiality reasons.
- **Class**: Target variable (1 for fraud, 0 for non-fraud).

## Data Exploration

The notebook begins by loading and exploring the dataset. Key steps include:
- Standardizing the `Time` and `Amount` features.
- Performing Welch’s t-tests to identify significant differences between fraudulent and non-fraudulent transactions.
- Visualizing the distribution of features using histograms.

## Data Partitioning

The dataset is partitioned into three subsets:
- **Training set**: 40% of the data.
- **Validation set**: 30% of the data.
- **Test set**: 30% of the data.

This partitioning ensures that the model is trained on one subset, validated on another, and finally tested on a separate subset to evaluate its performance.

## XGBoost Model Setup

The XGBoost model is set up with the following fixed parameters:
- **Objective**: Binary classification.
- **Evaluation metric**: Area Under Curve (AUC).
- **Number of boosting iterations**: 20.

The variable parameters to be tuned include:
- **max_depth**: Maximum depth of a tree.
- **subsample**: Proportion of training instances used in trees.
- **colsample_bytree**: Subsample ratio of columns.
- **eta**: Learning rate.
- **gamma**: Minimum loss reduction required for a split.
- **scale_pos_weight**: Controls the balance of positive and negative weights.

## Simulated Annealing for Hyper-parameter Tuning

Simulated Annealing is used to explore the hyper-parameter space efficiently. The algorithm works as follows:
1. Start with a random combination of hyper-parameters.
2. At each iteration, generate a neighboring combination of hyper-parameters.
3. If the new combination improves the model's performance (measured by F-Score), accept it.
4. If the new combination is worse, accept it with a probability that decreases over time (controlled by the "temperature" parameter).

The temperature starts high, allowing the algorithm to explore a wide range of hyper-parameters, and gradually decreases, focusing on the most promising regions.

## Results

The notebook tracks the F-Score on the validation set for each combination of hyper-parameters. The best combination found during the search is used to evaluate the model on the test set. The final model's performance is reported using a confusion matrix and the F-Score.

## Conclusion

This notebook demonstrates how to use XGBoost for credit card fraud detection and how to efficiently tune its hyper-parameters using Simulated Annealing. The approach is particularly useful for large datasets and complex models where exhaustive grid search is computationally prohibitive.

## Requirements

To run this notebook, you will need the following Python libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `xgboost`

You can install these libraries using pip:

```bash
pip install numpy pandas matplotlib scipy xgboost
```

## How to Run

1. Clone this repository.
2. Install the required libraries.
3. Open the Jupyter notebook `Aryan_bhardwaj_credit_card_fraud.ipynb`.
4. Run the notebook cells sequentially.

## Acknowledgments

- Dataset: [Credit Card Fraud Detection](https://www.kaggle.com/dalpozz/creditcardfraud) from Kaggle.
- XGBoost: [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/).

---

Feel free to contribute to this project by opening issues or pull requests. Happy coding! 🚀
