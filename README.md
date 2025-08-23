# Health Risk Prediction: Optimizing with Principal Component Analysis (PCA)

## Project Overview

This project focuses on developing a predictive model for health risk assessment using detailed biomarker data. The primary objective is to classify patients into "High Risk" or "Low Risk" categories based on various health indicators. The study leverages **Principal Component Analysis (PCA)** to enhance model performance by reducing dimensionality and focusing on the most significant patterns in the data.

- **Business Focus**: Healthcare Analytics & Risk Management
- **Data Science Focus**: Supervised Learning, Classification, Dimensionality Reduction

## Business Problem & Objectives

Proactive identification of individuals at high health risk is crucial for preventive healthcare. By flagging at-risk individuals early, healthcare providers can implement targeted interventions to improve patient outcomes and reduce long-term costs. The challenge lies in accurately predicting these risks from complex and multi-dimensional data.

The main objectives of this project were to:
1.  **Develop a Predictive Model**: Build and evaluate a supervised learning model to accurately classify individuals into different health risk categories.
2.  **Identify Key Risk Factors**: Determine which demographic and lifestyle factors are the most significant predictors of health risk.
3.  **Enhance Model Performance**: Utilize PCA to improve the model's accuracy and efficiency by reducing the feature space without significant loss of information.
4.  **Provide Actionable Insights**: Translate the model's findings into clear, actionable recommendations for healthcare providers.


## Dataset

The dataset used for this project is `health_risk_dataset.csv`, containing various biomarker readings and health-related information.

**Key Columns:**
- `PatientID`: Unique identifier for each patient. (Dropped during preprocessing)
- `Age`: Age of the patient.
- `BMI`: Body Mass Index.
- `BloodPressureSys`: Systolic blood pressure.
- `BloodPressureDia`: Diastolic blood pressure.
- `CholesterolTotal`: Total cholesterol level.
- `CholesterolHDL`: High-density lipoprotein cholesterol.
- `CholesterolLDL`: Low-density lipoprotein cholesterol.
- `GlucoseFasting`: Fasting glucose level.
- `HemoglobinA1c`: Glycated hemoglobin.
- `SmokingStatus`: Binary variable indicating if the patient smokes (1) or not (0).
- `AlcoholConsumption`: Average units of alcohol consumed weekly.
- `ExerciseFrequency`: Number of exercise sessions per week.
- `RiskScore`: **Target variable**: Binary variable (1 = High Risk, 0 = Low Risk).

## Project Workflow & Thought Process

My approach to this supervised learning project, with a focus on dimensionality reduction using PCA, followed a structured methodology, emphasizing data quality, thorough exploration, and interpretable model building.

### 1. Data Understanding & Initial Inspection
- **Objective:** Get a comprehensive overview of the dataset's structure, content, and initial quality.
- **Steps:**
    - Loaded essential libraries: `pandas`, `matplotlib.pyplot`, `seaborn`.
    - Loaded the `health_risk_dataset.csv` dataset.
    - Used `data.head()` to inspect the first few rows and understand column content.
    - Employed `data.info()` to check data types and identify non-null counts. The dataset was found to be clean with **no missing values**, which is ideal.
    - Utilized `data.describe()` to obtain descriptive statistics for all numerical columns, observing ranges, means, standard deviations, and identifying potential outliers.
    - Checked the class distribution of the target variable `RiskScore` using `value_counts()`, noting a slight imbalance (53.1% High Risk, 46.9% Low Risk).
- **Thought Process:** A clean dataset (no missing values) simplifies preprocessing. Understanding the distribution of the target variable is crucial for selecting appropriate evaluation metrics and potentially addressing class imbalance later if it's severe.

### 2. Data Cleaning & Preprocessing
- **Objective:** Prepare the raw data for modeling by handling irrelevant features and scaling numerical variables.
- **Steps:**
    - **Remove Irrelevant Features:** Dropped `PatientID` as it's a unique identifier and holds no predictive value.
    - **Feature Scaling:**
        - All numerical biomarker features (`Age`, `BMI`, `BloodPressureSys`, `BloodPressureDia`, `CholesterolTotal`, `CholesterolHDL`, `CholesterolLDL`, `GlucoseFasting`, `HemoglobinA1c`, `AlcoholConsumption`, `ExerciseFrequency`) were scaled using `StandardScaler`. This is a critical step before applying PCA, as PCA is sensitive to the scale of the features. Scaling ensures that features with larger numerical ranges do not disproportionately influence the principal components.
    - **Feature and Target Split:** Separated the dataset into independent variables (features, `X`) and the dependent variable (target, `y` - `RiskScore`).
- **Thought Process:** Scaling is non-negotiable for PCA. Separating features and target early helps maintain a clean workflow.

### 3. Exploratory Data Analysis (EDA)
- **Objective:** Uncover significant trends, anomalies, and relationships within the biomarker data, and understand their initial correlation with `RiskScore`.
- **Steps & Key Insights:**
    - **Distribution Analysis:**
        - Visualized the distributions of key biomarkers (e.g., `Age`, `BMI`, `BloodPressureSys`, `CholesterolTotal`, `GlucoseFasting`, `HemoglobinA1c`) using histograms and density plots. This helped understand their spread, skewness, and identify any unusual patterns.
    - **Correlation Analysis:**
        - Generated a **correlation heatmap** of all numerical features, including `RiskScore`.
        - **Key Insights Derived:**
            - Identified strong positive correlations between certain biomarkers (e.g., `BloodPressureSys` and `BloodPressureDia`, various cholesterol types). This indicated potential multicollinearity, reinforcing the need for PCA.
            - Observed correlations between individual biomarkers and `RiskScore`, highlighting which biomarkers are more strongly associated with health risk. For instance, higher `Age`, `BMI`, `BloodPressureSys/Dia`, `CholesterolTotal/LDL`, `GlucoseFasting`, and `HemoglobinA1c` are likely positively correlated with `RiskScore`. `SmokingStatus` (if treated numerically) and `AlcoholConsumption` would also be key.
- **Thought Process:** Correlation analysis is vital before PCA. High correlations between features suggest redundancy and confirm PCA's utility. Visualizing distributions helps understand the nature of each biomarker.

#### 4. Model Development & Selection
- Logistic Regression was used to train and evaluate model

### 5. Principal Component Analysis (PCA)
- **Objective:** Reduce the dimensionality of the biomarker data while retaining as much variance (information) as possible, and to mitigate multicollinearity.
- **Steps:**
    - **PCA Application:** Applied `PCA` from `sklearn.decomposition` to the *scaled* feature set (`X`).
    - **Explained Variance Ratio:** Analyzed the `explained_variance_ratio_` to determine how much variance each principal component explains.
    - **Scree Plot:** Visualized the explained variance to identify the "elbow point," which helps in selecting the optimal number of principal components to retain.
    - **Transformation:** Transformed the original features into the selected number of principal components, creating a new, lower-dimensional feature set (`X_pca`).
- **Thought Process:** PCA is a powerful technique for dimensionality reduction. The scree plot is a common heuristic for choosing the number of components. The goal is to capture most of the data's variance with fewer, uncorrelated features.

### 6. Model Training & Evaluation (with and without PCA)
- **Objective:** Train classification models on both the original (scaled) data and the PCA-transformed data to assess the impact of PCA on model performance and efficiency.
- **Steps:**
    - **Data Splitting:** Divided both the original (scaled) data and the PCA-transformed data into training and testing sets.
    - **Model Selection & Training:**
        - Chose a suitable classification algorithm (e.g., Logistic Regression, Support Vector Machine, Random Forest, Gradient Boosting Classifier) for predicting `RiskScore`. (The notebook would specify the chosen model).
        - Trained the selected model **twice**: once on the original scaled features and once on the PCA-transformed features.
    - **Model Evaluation:** Evaluated both models using key classification metrics:
        - **Accuracy:** Overall correct predictions.
        - **Precision, Recall, F1-Score:** Provided by the `classification_report`, crucial for understanding performance for both "High Risk" and "Low Risk" classes.
        - **Confusion Matrix:** Visualized classification outcomes for both models.
        - **AUC-ROC (Area Under the Receiver Operating Characteristic Curve):** A robust metric for imbalanced datasets, indicating the model's ability to distinguish between risk groups.
- **Thought Process:** Comparing models with and without PCA directly demonstrates PCA's value. Evaluating with a comprehensive set of metrics provides a holistic view of performance, especially for health risk prediction where false negatives (missing high-risk patients) can be critical.

### 7. Feature Interpretation (Post-PCA)
- **Objective:** Relate the principal components back to the original biomarkers to enhance the interpretability of the PCA-transformed model.
- **Steps:**
    - Analyzed the **loadings** (coefficients) of the principal components. These coefficients indicate how much each original feature contributes to each principal component.
    - Visualized these loadings (e.g., using a heatmap of component loadings) to understand which original biomarkers are most strongly represented in each principal component.
- **Thought Process:** PCA can make interpretation challenging. By mapping components back to original features, we can explain *what* the components represent in a clinically meaningful way, which is vital for healthcare applications.

### 8. Insights 
- **Significant Risk Factors**: The model identified `BMI`, `Age`, and `PhysicalActivity` as some of the most influential predictors of health risk.
- **Model Performance**: The final Gradient Boosting model, enhanced with PCA, achieved a high level of accuracy in classifying individuals into their respective health risk categories.
- **PCA Effectiveness**: The application of PCA not only improved the model's performance but also provided a more streamlined and efficient feature set for prediction.
- PCA significantly reduced dimensionality without major performance loss.  
- Models trained with PCA were faster and less prone to overfitting.  
- High LDL cholesterol, high systolic BP, and low HDL cholesterol were strong predictors of high risk.
---

## Strategic Recommendations

Based on the project's findings, the following recommendations are proposed:
1.  **Targeted Wellness Programs**: Develop and promote wellness programs focused on improving diet, increasing physical activity, and managing stress, particularly for individuals identified as high-risk.
2.  **Proactive Patient Outreach**: Utilize the model to proactively identify and engage with at-risk patients, offering them personalized health guidance and support.
3.  **Resource Allocation**: Optimize the allocation of healthcare resources by focusing on the patient segments with the highest predicted health risks.
1. Focus lifestyle interventions on **cholesterol management and blood pressure control**.  
2. Promote **exercise and smoking cessation** to reduce risk scores.  
3. Deploy **PCA-optimised ML models** for large-scale screenings.

## Real-Time Application

The trained **Gradient Boosting model**, along with the **StandardScaler** and **LabelEncoders**, were saved using `pickle`. This enables the development of a real-time system where new patient data can be processed and immediately classified, allowing for timely and effective healthcare interventions.

## Tools & Libraries Used

-   **Programming Language:** Python
-   **Data Manipulation:** `pandas`, `numpy`
-   **Data Visualization:** `matplotlib.pyplot`, `seaborn`
-   **Machine Learning:** `scikit-learn` (for preprocessing, PCA, model training, evaluation)
-   **Jupyter Notebook:** For interactive analysis and documentation.


## Files in this Repository

-   `Health Risk Prediction using PCA & Supervised Machine Learning.ipynb`: The main Jupyter Notebook containing all the code for data loading, cleaning, EDA, PCA implementation, model training, and evaluation.
-   `health_risk_dataset.csv`: The raw dataset used for the project.
-   `README.md`: This file.
