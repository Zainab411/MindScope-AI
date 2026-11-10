# MindScope AI

### AI-Based Mental Health Detection System Using NLP and Machine Learning

## 1. Project Overview

MindScope AI is a machine learning-based text classification system developed using Natural Language Processing (NLP) and Machine Learning (ML).

The system analyzes text entered by users and classifies it into predefined mental health-related categories based on patterns learned from the training dataset.

It provides classification results, prediction confidence, probability visualization and supportive messages through an interactive application.

## 2. Project Objectives

- Develop an automated text classification system using NLP and Machine Learning.
- Analyze user-provided text.
- Classify text into four predefined categories.
- Display prediction confidence and category-wise probability.
- Provide supportive messages based on prediction results.
- Develop interactive and user-friendly application interfaces.

## 3. Classification Categories

The system classifies user-provided text into four categories:

1. **Normal:** Text indicating generally normal emotional patterns.
2. **Anxiety:** Text containing patterns associated with anxiety.
3. **Depression:** Text containing patterns associated with depression.
4. **Suicidal:** Text containing patterns associated with suicidal thoughts.

## 4. Sample Inputs

The following examples show the type of text that can be entered into the system.

| Category | Sample Input |
|---|---|
| Normal | "I had a wonderful day and feel happy." |
| Anxiety | "I feel nervous and keep worrying about everything." |
| Depression | "I have been feeling low and losing interest in things." |
| Suicidal | "I keep having disturbing thoughts about hurting myself." |

*Note: These are only examples. Actual predictions may vary depending on the trained model.*

## 5. Technologies Used

- **Programming Language:** Python
- **Machine Learning Library:** Scikit-learn
- **Natural Language Processing:** TF-IDF (Term Frequency–Inverse Document Frequency)
- **Data Processing:** Pandas
- **Data Visualization:** Matplotlib
- **Application Frameworks:** Streamlit and Flask
- **Frontend Technologies:** HTML, CSS and JavaScript
- **Model Saving:** Pickle

## 6. Machine Learning Workflow

1. **Dataset Loading:** Load the mental health text dataset.
2. **Data Preprocessing:** Clean and prepare the text data for training.
3. **Text Vectorization:** Convert text into numerical features using TF-IDF.
4. **Model Training:** Train a Logistic Regression classification model.
5. **Model Evaluation:** Evaluate the model using accuracy, F1-score, classification report and confusion matrix.
6. **Model Saving:** Save the trained model and TF-IDF vectorizer using Pickle.
7. **Text Prediction:** Use the saved model to classify newly entered text.

## 7. Key Features

- NLP-based mental health text classification.
- Four predefined classification categories.
- Logistic Regression-based predictions.
- Prediction confidence percentage.
- Category-wise probability visualization.
- Supportive messages based on predicted categories.
- Severity indication.
- Recent analysis history.
- Safety guidance for predictions associated with suicidal thoughts.
- Interactive application interfaces.
- Custom-designed Streamlit interface with a chocolate and beige color theme.

## 8. Application Interfaces

MindScope AI provides two separate application interfaces.

### 8.1 Streamlit Application

The Streamlit application provides an interactive interface where users can:

- Enter text for analysis.
- View the predicted category.
- Check prediction confidence.
- View category-wise probability charts.
- Receive supportive messages.
- View recent analysis history.
- Access safety guidance when applicable.

**To run the Streamlit application:**

```bash
python -m streamlit run streamlit_app.py
```

### 8.2 Flask Application

The Flask application provides a web-based interface developed using Flask, HTML, CSS and JavaScript.

It includes:

- Text input and analysis.
- Classification results.
- Prediction confidence and severity information.
- Supportive messages.
- Probability visualization.
- Recent analysis history.

**To run the Flask application:**

```bash
python app.py
```

Both applications work independently and provide separate interfaces for the same project.

## 9. Project Structure

```text
MindScope-AI/
│
├── static/
│   ├── script.js
│   └── style.css
│
├── templates/
│   └── index.html
│
├── app.py
├── streamlit_app.py
├── train_model.py
├── mental_health.csv
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
└── README.md
```

## 10. Model Evaluation

The trained machine learning model is evaluated using the following metrics:

- **Accuracy Score:** Measures how many predictions are correct.
- **F1-Score:** Measures the balance between precision and recall.
- **Classification Report:** Shows precision, recall and F1-score for each category.
- **Confusion Matrix:** Shows correct and incorrect predictions for each category.

These metrics help evaluate the performance of the classification model.

## 11. Important Note

MindScope AI is an educational Artificial Intelligence and Machine Learning project. It is a preliminary text classification prototype that identifies patterns in user-provided text.

Its predictions are not medical diagnoses and should not replace professional mental health assessment or support.

## 12. Project Information

- **Project Name:** MindScope AI
- **Project Domain:** Artificial Intelligence and Machine Learning
- **Project Type:** NLP-Based Text Classification System
- **Programming Language:** Python
- **Core Technologies:** NLP, Machine Learning, Scikit-learn, Streamlit and Flask