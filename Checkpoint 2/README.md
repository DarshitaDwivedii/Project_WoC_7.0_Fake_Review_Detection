# 🚀 Checkpoint 2: Model Training for Fake Review Detection

---

## 📂 What's in This Folder?

This folder contains the code and resources for **model training** in the Fake Review Detection project. Here's what you'll find:

### 📋 Contents:
- `model_training.py`: The main Python script for training and evaluating multiple machine learning models.
- `random_forest_model.pkl`: Serialized Random Forest model for later use.
- `svm_model.pkl`: Serialized Support Vector Machine (SVM) model for later use.
- `logistic_regression_model.pkl`: Serialized Logistic Regression model for later use.

---

## 🛠️ What Does the Code Do?

1. **Dataset Preparation** 🗂️
   - Loads the preprocessed dataset from Checkpoint 1.
   - Splits the data into training and testing sets.

2. **Model Training** 🤖
   - Trains the following models:
     - Random Forest
     - Support Vector Machine (SVM)
     - Logistic Regression
   - Uses pipelines for preprocessing and classification.

3. **Model Evaluation** 📊
   - Evaluates models using metrics like:
     - **Accuracy**
     - **Precision**
     - **Recall**
     - **F1 Score**
   - Compares the performance of all models.

4. **Model Serialization** 💾
   - Saves the trained models as `.pkl` files for future use.

---

## 🧪 Results

Each model is evaluated, and the results (Accuracy, Precision, Recall, F1 Score) are logged to compare performance and identify the best-performing model.

