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
![Methodology Diagram](samples/methodology.png)

---

## Project Structure

