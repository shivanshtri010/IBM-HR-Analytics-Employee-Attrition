# Employee Attrition Analysis

This code performs an analysis on the "WA_Fn-UseC_-HR-Employee-Attrition.csv" dataset to predict employee attrition. The code includes data preprocessing, exploratory data analysis, feature engineering, and model evaluation using various machine learning algorithms.

## Requirements

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Imbalanced-learn

## Usage

1. Install the required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
```

2. Download the "WA_Fn-UseC_-HR-Employee-Attrition.csv" dataset and place it in the same directory as the code file. [Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

3. Run the code file in your Python environment.

The code will perform the following steps:

- Load the dataset and perform initial data inspection
- Conduct exploratory data analysis and visualizations
- Perform feature engineering and data preprocessing
- Evaluate baseline models using various metrics
- Tune model hyperparameters for improved performance
- Compare baseline and tuned model performances
- Visualize performance metrics and ROC curves
- Apply oversampling techniques (SMOTE) to handle class imbalance
- Re-evaluate models on the oversampled data

## Code Overview

The code consists of the following main sections:

1. **Data Loading and Inspection**: The dataset is loaded, and initial data inspection is performed.
2. **Exploratory Data Analysis**: Visualizations and statistical analyses are conducted to understand the data and identify patterns.
3. **Feature Engineering**: New features are created based on domain knowledge and data insights.
4. **Data Preprocessing**: Categorical variables are label encoded, and unnecessary features are dropped.
5. **Baseline Model Evaluation**: Various machine learning models are evaluated using cross-validation and different performance metrics.
6. **Hyperparameter Tuning**: Model hyperparameters are tuned using cross-validation to improve performance.
7. **Performance Comparison**: Baseline and tuned model performances are compared and visualized.
8. **Oversampling**: The SMOTE technique is applied to handle class imbalance, and models are re-evaluated on the oversampled data.

## Additional Information

- The code provides detailed comments and explanations for each step.
- Visualizations are used extensively to aid in data understanding and model evaluation.
- The code can be modified to include additional models, feature engineering techniques, or evaluation metrics as needed.


