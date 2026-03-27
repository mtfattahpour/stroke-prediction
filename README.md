# Stroke Prediction

This project tackles the clinical challenge of predicting stroke risk, to help identify patients at higher risk than others, to deploy more effective preventive measures (more selective lifestyle correction and risk factor reduction efforts), using the [Stroke Prediction Dataset from Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset). My primary focus was not just on achieving a high accuracy score, but on overcoming the core methodological hurdles inherent in working with real-world clinical data; most notably, severe class imbalance.

In a clinical context, failing to identify a patient at risk (a false negative) is far more costly than flagging a healthy patient for a follow-up (a false positive). Therefore, this entire analysis is optimized for **recall** (sensitivity), with the goal of building a model that can reliably identify potential stroke cases.

## The Investigative Workflow

The project is structured across three Jupyter notebooks, each representing a distinct phase of the analysis, to create a clear and reproducible research path.

1.  notebooks/data_exploration.ipynb:
- A thorough cleaning of the data, including handling missing BMI values and outliers.
- Conducting a comprehensive exploratory data analysis (EDA) to understand feature distributions and their initial correlations with stroke.
- Rigorously validating visual insights using statistical tests.
- Concluded by building a baseline Random Forest model, which achieves 95% accuracy but a recall of 0, powerfully demonstrating that naive modeling is clinically useless on this imbalanced dataset.

2. notebooks/preprocessing.ipynb:
- This notebook serves as the core experimental phase.
- Developed a rich set of new features based on domain knowledge, including interaction terms (age*bmi), risk scores, and categorical binning of continuous variables (age_group, bmi_category).
- Systematically explored strategies to negate class imbalance, ultimately settling on a combined approach: mild oversampling with **SMOTE** and the use of **class_weight** parameters within the models.

3. notebooks/models.ipynb:
- This notebook operationalizes the findings from the previous phases.
- Applies the full preprocessing pipeline to the data.
- Conducts a head-to-head comparison of multiple classifiers, using cross-validation optimized for recall.
- Performs an extensive hyperparameter search (GridSearchCV) on the top-performing model (XGBoost) to maximize its predictive power.

## Key Findings and Final Model

*  The Recall/Precision Trade-Off: The final, tuned **XGBoost model** successfully navigates the class imbalance, achieving a **recall of 84%** on the test set. This means it correctly identifies 84% of true stroke cases.
*  Clinical Interpretation: This high recall comes at the cost of precision (12%). While this means a high number of false positives, in a screening context, this is an acceptable trade-off. The model is effective as a tool to flag high-risk individuals for further, more detailed clinical assessment.
*  Feature Importance: The most predictive features in the final model were not the raw inputs, but the engineered ones. `age_group_65+`, the custom `health_risk_score`, and `smoking history` were consistently the most powerful predictors, validating the feature engineering effort.
*  The "Formerly Smoked" Anomaly: A recurring, counter-intuitive finding was the high predictive importance of the formerly smoked status. Also, the normal BMI apparently contributed more than other categories of BMI to model performance.

## Discussion
SMOTE helps, but its synthetic samples don't fully replace real stroke cases. Collecting more data especially from under-represented age and risk groups could help considerably. Log-transforming skewed features like avg_glucose_level might also squeeze out a bit more performance. Finally optimizing the classification threshold via the ROC curve rather than using the default 0.5 cutoff may also be a straightforward improvement that could further boost recall without sacrificing too much precision.

## How to Run This Project

1.  Clone the repository.
2.  Set up a virtual environment and install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
4.  Follow the notebooks in order, starting with data_exploration.ipynb, to trace the full analytical journey.