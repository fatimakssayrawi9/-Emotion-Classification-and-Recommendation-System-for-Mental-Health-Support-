# Notebooks Folder

This folder contains all the Jupyter notebooks used to experiment with various datasets and models for emotion classification. Each notebook is focused on a specific experiment or model evaluation.

## Contents

### 1. **ISEAR_Dataset.ipynb**
- **Purpose**: Evaluate classical machine learning models on the balanced ISEAR dataset with seven emotions.
- **Dataset Used**: ISEAR dataset.
- **Description**: Demonstrates the poor performance of traditional machine learning models (Logistic Regression, Naive Bayes, Decision Tree, Random Forest, Ensemble Classifier) on emotion classification, even with balanced data.

### 2. **GoEmotions_Dataset.ipynb**
- **Purpose**: Clean and preprocess the GoEmotions dataset to match the format of the ISEAR dataset.
- **Dataset Used**: GoEmotions dataset.
- **Description**: Focuses on transforming multi-label samples into single-label entries and reducing emotion categories from 28 to 11.

### 3. **Combined_Dataset_11_emotions_Classical_ML_Without_Class_Imbalance.ipynb**
- **Purpose**: Train classical models on the combined ISEAR and GoEmotions dataset without handling class imbalance.
- **Dataset Used**: Combined ISEAR and GoEmotions datasets.
- **Description**: Reveals the performance drop in minority classes when class imbalance is not addressed.

### 4. **Combined_Dataset_11_emotions_Classical_ML_With_Class_Imbalance_SMOTE.ipynb**
- **Purpose**: Handle class imbalance using SMOTE and evaluate classical models.
- **Dataset Used**: Combined ISEAR and GoEmotions datasets (balanced using SMOTE).
- **Description**: It didn't show improved performance for minority classes while maintaining overall accuracy.

### 5. **Combined_Dataset_11_emotions_Classical_ML_With_Class_Imbalance_Class_Weighting.ipynb**
- **Purpose**: Address class imbalance by applying class weighting during model training.
- **Dataset Used**: Combined ISEAR and GoEmotions datasets.
- **Description**: It didn't demonstrates improvement in minority class metrics with weighted models.

### 6. **Handle_Class_Imbalance_Data_Augmentation_Synonyms_Approach.ipynb**
- **Purpose**: Augment the dataset using synonym replacement to address class imbalance.
- **Dataset Used**: Combined ISEAR and GoEmotions datasets.
- **Description**: Creates an augmented dataset with balanced emotion classes.

### 7.a **Augmented_Dataset_Classical_ML_Train_Test.ipynb**
- **Purpose**: Evaluate classical models (Logistic Regression, Naive Bayes, Decision Tree, Random Forest) on the augmented dataset.
- **Dataset Used**: Augmented dataset.
- **Description**: Shows improvements in minority class performance using the augmented dataset.

### 7.b **Augmented_Dataset_Ensemble_Classifiers.ipynb**
- **Purpose**: Train ensemble classifiers (AdaBoost, Gradient Boosting) on the augmented dataset.
- **Dataset Used**: Augmented dataset.
- **Description**: It didn't demonstrate better generalization and handling of imbalanced data.

### 8.a **Augmented_Dataset_Logistic_Regression_Train_Val_Test.ipynb**
- **Purpose**: Train Logistic Regression with train/validation/test splits as a baseline model.
- **Dataset Used**: Augmented dataset.
- **Description**: Establishes a baseline with moderate performance.

### 8.b **Augmented_Dataset_Logistic_Regression_Hyperparameter_Tuning.ipynb**
- **Purpose**: Optimize Logistic Regression hyperparameters (e.g., regularization strength, solver) using GridSearchCV.
- **Dataset Used**: Augmented dataset.
- **Description**: Improves baseline performance a little bit, particularly for minority classes.

### 9.a **Augmented_Dataset_Naive_Bayes_Train_Val_Test.ipynb**
- **Purpose**: Train and evaluate Naive Bayes on train/validation/test splits.
- **Dataset Used**: Augmented dataset.
- **Description**: Provides a baseline analysis for Naive Bayes.

### 9.b **Augmented_Dataset_Naive_Bayes_Hyperparameter_Tuning_Cross_Validation.ipynb**
- **Purpose**: Tune Naive Bayes' smoothing parameter (alpha) using GridSearchCV.
- **Dataset Used**: Augmented dataset.
- **Description**: Improves Naive Bayes' performance a little bit compared to the baseline.

### 10.a **Augmented_Dataset_Decision_Tree_Train_Val_Test.ipynb**
- **Purpose**: Train and evaluate a Decision Tree on train/validation/test splits.
- **Dataset Used**: Augmented dataset.
- **Description**: Assesses Decision Tree's baseline performance.

### 10.b **Augmented_Dataset_Decision_Tree_Hyperparameter_Tuning_Cross_Validation.ipynb**
- **Purpose**: Optimize Decision Tree hyperparameters (e.g., splitting criterion, depth) using GridSearchCV.
- **Dataset Used**: Augmented dataset.
- **Description**: Improves performance a little bit compared to the baseline.

### 11.a **Augmented_Dataset_Random_Forest_Train_Val_Test.ipynb**
- **Purpose**: Train and evaluate Random Forest on train/validation/test splits.
- **Dataset Used**: Augmented dataset.
- **Description**: Provides a baseline analysis for Random Forest.

### 11.b **Augmented_Dataset_Random_Forest_Hyperparameter_Tuning_Cross_Validation_Train_Val_Test.ipynb**
- **Purpose**: Optimize Random Forest hyperparameters (e.g., estimators, depth) using GridSearchCV.
- **Dataset Used**: Augmented dataset.
- **Description**: Enhances performance, particularly for minority classes.

### 12.a **Augmented_Dataset_Stacking_Classifier_Base_Models.ipynb**
- **Purpose**: Train a stacking classifier using Logistic Regression, Naive Bayes, Decision Tree, and Random Forest as base models, with Logistic Regression as the meta-model.
- **Dataset Used**: Augmented dataset.
- **Description**: Combines multiple models for robust performance.

### 12.b **Augmented_Dataset_Stacking_Classifier_Hyperparameter_Tuning.ipynb**
- **Purpose**: Optimize base models and meta-model within the stacking classifier using GridSearchCV.
- **Dataset Used**: Augmented dataset.
- **Description**: Shows small improvements across all metrics with optimized stacking.

### 13. **Augmented_Dataset_CNN.ipynb**
- **Purpose**: Preliminary exploration of a Convolutional Neural Network (CNN) for emotion classification.
- **Dataset Used**: Augmented dataset.
- **Description**: Demonstrates potential for deep learning approaches in emotion classification, with promising results.

## Notes
- Each notebook is self-contained and provides insights into the respective experiments and models.
- Ensure all dependencies (e.g., scikit-learn, TensorFlow) are installed before running the notebooks.
- Upload the corresponding dataset for each notebook.

