# Models Folder

This folder contains the trained machine learning models and supporting vectorizers for emotion classification. Due to the size of the models, the actual files are hosted externally, and links to download them are provided below.

## Contents

### 1. **TF-IDF Vectorizer**
- **Description**: The trained TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer used for converting text data into numerical representations suitable for machine learning models.
- **Link**: [Download the TF-IDF Vectorizer](<insert-google-drive-link-here](https://drive.google.com/file/d/1au-rofToOgk8nzLnVoXhWFMNu9zKGfNr/view?usp=sharing)>)
- **Purpose**:
  - Converts raw text into a sparse matrix of TF-IDF features.
  - Ensures consistent text processing for the trained stacking classifier.

### 2. **Stacking Classifier**
- **Description**: The trained stacking classifier with the best performance compared to other models in the project. 
- **Key Features**:
  - Combines multiple classifiers into a meta-classifier for improved performance.
  - Uses the TF-IDF vectorizer as input for predictions.
  - Handles class imbalance effectively due to data augmentation and weighted learning techniques.
- **Size**: 1.5 GB
- **Link**: [Download the Stacking Classifier](<https://drive.google.com/file/d/1rgNxtmpNlTDEwbhoV2Lfibxlrd-bQ0K7/view?usp=sharing>)
- **Performance**:
  - Outperformed individual models (e.g., Random Forest, Gradient Boosting).
  - Achieved the highest F1 score and accuracy on the test dataset.

## Notes
- **Usage**:
  - Load the TF-IDF vectorizer first to transform input text data.
  - Use the stacking classifier for predictions.
- **File Structure**:
  - The vectorizer and model are stored as serialized files (`.pkl` or similar format) for efficient loading.
- **Dependencies**:
  - Ensure all dependencies are installed (e.g., scikit-learn, joblib) before using the vectorizer and classifier.
- **Memory Requirements**:
  - Ensure sufficient memory and disk space to handle the model's size (1.5 GB).

## Instructions to Load the Model and Vectorizer

### Loading the TF-IDF Vectorizer
```python
import joblib

# Load the vectorizer
vectorizer = joblib.load("path_to_vectorizer.pkl")

# Transform input text
text_data = ["Sample sentence for classification"]
features = vectorizer.transform(text_data)