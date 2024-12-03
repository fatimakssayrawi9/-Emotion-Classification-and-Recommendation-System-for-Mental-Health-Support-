# Source Code (src) Folder

This folder contains the code for deploying the emotion classification system using the best-performing model (Stacking Classifier) and an additional feature for recommending mental health resources.

## Contents

### 1. **Emotion_Classification_Deployment.ipynb**
- **Purpose**: Deploy the emotion classification system and recommend mental health resources based on detected emotions.
- **Model Used**: Stacking Classifier (joblib format, trained version).
- **Features**:
  1. Emotion classification using text input.
  2. Mental health resource recommendations based on the detected emotion.
- **Dataset**: Preprocessed dataset used to train the best model (Stacking Classifier).
- **Usage**:
  - Simply run the notebook in Google Colab (no file uploads required).
  - Enter your emotion or a sentence describing your feelings.
  - The system will classify your emotion and provide resource recommendations.

## How to Use
1. Open the notebook in Google Colab.
2. Downolad the Stacking Classifier model and TfIDf Vectorizer from Google Drive using the links provided in the README.md file of the **models** folder and upload them into the Colab Notebook
3. Create an API key and replace it in its placeholder (if you don't have one, I can share with u my secret key to run the code)
4. Run all cells in the notebook.
5. When prompted, enter a text input describing your feelings (e.g., "I feel so anxious about work").
6. The system will:
   - Predict your emotion.
   - Recommend mental health resources tailored to your emotion.

## Sample Run
Below is an example of a sample run:

![Sample Run](Joy.png)


