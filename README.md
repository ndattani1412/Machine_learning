# Machine Learning Projects Portfolio

## **Project 1**  
### Project Title:  
**Predicting Customer Satisfaction for Santander Bank Using Decision Tree Models**

### Project Description:  
In this project, I aimed to identify dissatisfied customers early in their journey with Santander Bank using machine learning techniques. Leveraging the anonymized dataset provided by Kaggle, I built and evaluated classification models to distinguish between satisfied (`TARGET=0`) and dissatisfied (`TARGET=1`) customers.

### Key Steps:
1. **Data Loading & Cleaning**  
   - Loaded both training and test datasets from the provided files.  
   - Handled missing values in the test dataset using column-wise mean imputation to maintain data integrity.

2. **Feature & Target Separation**  
   - Separated the features (`X`) and the target variable (`y`) from the training set.

3. **Train-Test Split**  
   - Split the dataset into training and validation sets using an 80-20 split to assess model performance.

4. **Baseline Modeling**  
   - Trained a baseline Decision Tree Classifier and evaluated its accuracy on the validation set.  
   - Explored the impact of class balancing using the `class_weight='balanced'` parameter.

5. **Model Tuning & Comparison**  
   - Built and tested multiple decision tree models with different hyperparameters:  
     - `max_depth=10`  
     - `criterion='gini'`  
     - `max_leaf_nodes=15`  
   - Compared their validation accuracies to select the best-performing configuration.

6. **Prediction & Submission (commented out)**  
   - Prepared logic to make predictions on the test dataset and create a Kaggle submission file, though this was commented out in the final notebook.

### Technologies Used:
- Python, Pandas  
- Scikit-learn (`DecisionTreeClassifier`, `train_test_split`, `accuracy_score`)

### Outcomes:
- Gained hands-on experience tuning decision tree models for classification.  
- Identified model parameters that improve predictive performance on imbalanced classes.  
- Built a foundational pipeline for customer satisfaction prediction using real-world business data.

---

## **Project 2**  
### Project Title:  
**Predicting Insurance Quote Conversion Using Ensemble and Neural Network Models for Homesite Insurance**

### Project Description:  
This project focuses on predicting the likelihood of a customer purchasing a home insurance policy based on a quote, using an anonymized dataset provided by Homesite Insurance. Accurately predicting conversion rates empowers the company to fine-tune pricing strategies and better target potential customers, improving their overall sales pipeline efficiency and customer segmentation.

### Key Objectives:
- Build a classification model to predict the `QuoteConversion_Flag`.  
- Handle data preprocessing, feature scaling, class imbalance, and model selection.  
- Evaluate and compare multiple models, including ensemble methods and neural networks.

### Key Steps:
1. **Data Loading and Exploration**  
   - Loaded training and testing datasets.  
   - Explored data structure and checked for missing values and class distribution.

2. **Preprocessing**  
   - Addressed class imbalance using **SMOTE**.  
   - Applied **StandardScaler** for normalization.  
   - Handled missing values using `SimpleImputer`.

3. **Feature Engineering and Splitting**  
   - Separated input features and target variable.  
   - Performed stratified split (80/20) for training and validation.

4. **Model Development and Tuning**  
   - Trained multiple classifiers:
     - Random Forest  
     - Support Vector Machine (SVM)  
     - Decision Tree  
     - K-Nearest Neighbors  
     - MLP Classifier (Neural Network)  
     - Stacking Classifier (Ensemble)  
   - Used `GridSearchCV` for hyperparameter tuning.

5. **Model Evaluation**  
   - Evaluated models using:
     - Classification Report (Precision, Recall, F1-score)  
     - ROC AUC Score

6. **Testing and Final Predictions**  
   - Scaled the test dataset and prepared prediction pipeline.  
   - Integrated final model for deployment.

### Technologies Used:
- Python, Pandas, NumPy  
- Scikit-learn (`MLPClassifier`, `RandomForestClassifier`, `GridSearchCV`, `StackingClassifier`)  
- Imbalanced-learn (`SMOTE`)  
- StandardScaler, Classification metrics

### Outcomes:
- Successfully built a machine learning pipeline capable of handling class imbalance and feature scaling.  
- Identified optimal models with robust classification metrics.  
- Demonstrated the potential of ensemble and neural models in quote conversion prediction.  
- Created a scalable solution Homesite could use to estimate the impact of pricing strategies.

---

## **Project 3**  
### Project Title:  
**Forecasting Russian Real Estate Prices Using Feature-Rich Regression Models for Sberbank**

### Project Description:  
This project tackles the challenge of predicting apartment sale prices in Russia using a blend of housing characteristics and macroeconomic indicators. Conducted as part of the Sberbank Russian Housing Market Kaggle competition, the goal was to build robust regression models that account for volatile economic conditions, enabling Sberbank to provide price guidance to developers, lenders, and homebuyers with improved certainty.

### Key Objectives:
- Predict apartment sale prices (`price_doc`) using property and economic features.  
- Incorporate temporal and macroeconomic trends.  
- Select the most relevant features to enhance model generalization.

### Approach & Key Steps:
1. **Data Integration & Enrichment**  
   - Merged real estate data with macroeconomic indicators on timestamps.  
   - Extracted additional date-based features like year, month, and day of the week.

2. **Preprocessing**  
   - Imputed missing values using `SimpleImputer`.  
   - Standardized features with `StandardScaler`.  
   - Encoded categorical variables using `LabelEncoder`.

3. **Feature Selection Techniques**  
   - Applied `SelectFromModel`, `Sequential Feature Selector`, and **Genetic Algorithms**.  
   - Compared model performance with and without feature selection.

4. **Model Development**  
   - Trained and tuned multiple regression models:
     - LightGBM  
     - Random Forest Regressor  
     - Gradient Boosting Regressor  
     - Support Vector Regressor (SVR)  
     - Ridge Regression  
   - Used cross-validation and `GridSearchCV`.

5. **Model Evaluation**  
   - Used **Root Mean Squared Logarithmic Error (RMSLE)** as the evaluation metric.  
   - Compared pipelines to determine the best-performing model.

### Technologies Used:
- Python, Pandas, NumPy, Matplotlib  
- Scikit-learn, LightGBM  
- Genetic Algorithms (for feature optimization)  
- Feature Selection: `SelectFromModel`, `SequentialFeatureSelector`  
- Evaluation Metric: RMSLE

### Outcomes:
- Built a robust regression pipeline integrating macro and housing data.  
- Improved generalization by optimizing and reducing feature space.  
- Demonstrated ML's value in supporting real estate pricing amid economic volatility.

---
