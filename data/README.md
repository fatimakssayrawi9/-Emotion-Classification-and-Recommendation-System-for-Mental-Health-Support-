# Data Folder

This folder contains all the datasets used for the emotion analysis project. These datasets are either raw or preprocessed to prepare them for model training and evaluation.

## Datasets Included

### 1. **ISEAR Dataset**
- **Description**: The International Survey on Emotion Antecedents and Reactions (ISEAR) dataset, containing emotional text samples labeled with seven emotions.
- **Columns**:
  - **Emotion**: Target label for one of seven emotions: joy, fear, sadness, anger, shame, disgust, or guilt.
  - **Text**: Sentence expressing the associated emotion.
- **Notes**:
  - Original raw dataset without preprocessing.
  - Approximately equal distribution of all emotion classes.

### 2. **ISEAR Cleaned**
- **Description**: A preprocessed version of the ISEAR dataset.
- **Preprocessing Steps**:
  - Converted text to lowercase.
  - Removed special characters and stop words.
  - Applied stemming to reduce words to their base form.
- **Purpose**: Used to improve consistency and feature extraction for training.

### 3. **Go Emotions (11 Emotions Cleaned)**
- **Description**: A cleaned version of the GoEmotions dataset by Google, containing emotional text mapped to 11 categories.
- **Characteristics**:
  - Originally consisted of 28 emotion categories, merged into 11 emotions to align with the ISEAR dataset.
  - Includes new categories: neutral, surprise, confusion, and boredom.
- **Columns**:
  - **Text**: Sentence expressing an emotion.
  - **Label**: The most frequent single emotion label for each sample.
- **Purpose**: Merged with the ISEAR dataset for creating a unified dataset.

### 4. **Combined Emotion Dataset**
- **Description**: A dataset created by combining the cleaned ISEAR and GoEmotions datasets.
- **Characteristics**:
  - Contains 11 emotional categories.
  - Imbalanced dataset but cleaned.
- **Purpose**: Provides a larger, unified dataset for emotion classification.

### 5. **Augmented Emotion Dataset**
- **Description**: A final dataset that addresses class imbalance through data augmentation techniques.
- **Key Features**:
  - Synonym-based augmentation using WordNet for underrepresented classes.
  - Balanced representation of all 11 emotions.
  - Includes additional samples for rare emotions like surprise and boredom.
- **Purpose**: Mitigates class imbalance, ensuring effective training for machine learning models.

## Notes
- The datasets are saved as `.csv` files for easy loading and manipulation.
- Ensure appropriate preprocessing steps are applied based on the dataset selected for use.
- The augmented dataset is the most balanced and is recommended for model training.

## Usage
- Use the `ISEAR_dataset.csv` for the notebook 1
- Use the `Go_Emotions_11_Emotions_Cleaned.csv` for 3
- Use the `ISEAR_Cleaned.csv` for the notebook 3
- Use the `Combined_Emotion_Dataset.csv` for the notebooks 4, 5, 6
- Use the `Augmented_Emotion_Dataset.csv` for the notebooks 7(a,b), 8(a,b), 9(a,b), 10(a,b), 11(a,b), 12(a,b), 13 
