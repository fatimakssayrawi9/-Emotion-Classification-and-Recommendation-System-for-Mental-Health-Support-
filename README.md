# Emotion Classification and Recommendation System for Mental Health Support

## Introduction
This project focuses on emotion classification using text data and provides recommendations for mental health resources based on the detected emotion. The system is designed to process textual inputs, classify the emotions expressed, and suggest relevant resources to improve emotional well-being. The project leverages a combination of advanced machine learning techniques and publicly available datasets.

The core of the project is a **Stacking Classifier** that combines multiple classical models, delivering robust performance. The model was trained on an augmented dataset combining the ISEAR and GoEmotions datasets.

---

## Final Model and Dataset
### **Final Model:**
- **Stacking Classifier**:
  - Base Models: Logistic Regression, Naive Bayes, Decision Tree, Random Forest.
  - Meta-Model: Logistic Regression.
  - Training and validation were performed using hyperparameter tuning to achieve optimal results.
  - The final model was saved using **joblib** for efficient deployment.

### **Dataset Used:**
1. **ISEAR Dataset**:
   - Contains text samples labeled with seven emotions (e.g., joy, fear, anger).
   - Balanced and preprocessed for training.

2. **GoEmotions Dataset**:
   - Originally multi-label and with 28 emotions.
   - Transformed into 11 emotions and integrated with ISEAR.

3. **Augmented Dataset**:
   - Combines the cleaned ISEAR and GoEmotions datasets.
   - Addressed class imbalance using  **synonym-based augmentation techniques**.
   - Includes 11 emotions:
     - Joy, Sadness, Anger, Fear, Shame, Disgust, Guilt, Neutral, Surprise, Confusion, and Boredom.

---

## Methodology
The project follows the following methodology to achieve emotion classification and provide recommendations:

1. **Data Collection and Preprocessing**:
   - Collected and cleaned text data from ISEAR and GoEmotions datasets.
   - Addressed class imbalance using techniques like SMOTE and synonym-based augmentation.

2. **Model Selection**:
   - Experimented with classical machine learning models (Logistic Regression, Naive Bayes, etc.).
   - Finalized the Stacking Classifier for deployment based on superior performance metrics.

3. **Model Deployment**:
   - The best model and TF-IDF vectorizer were saved using **joblib**.
   - The deployment system uses **OpenAI API** for recommending mental health resources based on detected emotions.

### Methodology Diagram:
![Methodology Diagram](Methodology.png)

---

## Project Structure
```plaintext
emotion-classification/
├── data/
│   ├── ISEAR_dataset.csv                  # Raw ISEAR dataset
│   ├── ISEAR_cleaned.csv                  # Preprocessed ISEAR dataset
│   ├── GoEmotions_11_Emotions_Cleaned.csv # Cleaned GoEmotions dataset
│   ├── Combined_Emotion_Dataset.csv       # Combined dataset of ISEAR and GoEmotions
│   ├── Augmented_Emotion_Dataset.csv      # Final augmented dataset
│   └── README.md                          # Description of datasets
├── models/
│   ├── stacking_model.pkl                 # Trained stacking classifier model
│   ├── tfidf_vectorizer.pkl               # TF-IDF vectorizer used for model input
│   └── README.md                          # Description of models and usage
├── notebooks/
│   ├── 1.ISEAR_Dataset.ipynb              # Analysis and experiments on the ISEAR dataset
│   ├── 2.GoEmotions_Dataset.ipynb         # Preprocessing and cleaning GoEmotions dataset
│   ├── 3.Combined_Dataset_Classical_ML.ipynb # Experiments using classical models on combined dataset
│   ├── 4.Augmented_Dataset_SMOTE.ipynb    # Handling imbalance using SMOTE and data augmentation
│   ├── 5.Stacking_Classifier_Deployment.ipynb # Experiments with stacking classifier
│   └── README.md                          # Description of notebooks
├── src/
│   ├── Emotion_Classification_Deployment.ipynb # Deployment code for emotion classification
│   ├── samples/
│   │   ├── Boredom.png                    # Example output for Boredom emotion
│   │   └── methodology.png                # Methodology diagram for the project
│   └── README.md                          # Description of deployment process
├── README.md                              # Main project description
└── requirements.txt                       # List of required dependencies
