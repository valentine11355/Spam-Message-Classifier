# 📩 Spam Message Classifier

## Overview
This project classifies SMS messages as Spam or Not Spam using Machine Learning.

## Model
- TF-IDF Vectorizer
- Linear SVM (High Accuracy ~98–99%)

## Tech Stack
- Python
- pandas
- scikit-learn
- Streamlit

## Features
- Real-time spam detection
- Clean UI with Streamlit
- High accuracy

## Screenshot
<img width="1912" height="662" alt="Screenshot 2026-04-03 151927" src="https://github.com/user-attachments/assets/60fc8bef-a746-4e4a-83ef-c892676ff846" />
<img width="1915" height="617" alt="Screenshot 2026-04-03 151805" src="https://github.com/user-attachments/assets/35ea04ff-6f46-4945-a48e-ab3a7b1db97c" />

## Approach
- I built a Spam Message Classifier using a simple Machine Learning pipeline.
- First, I cleaned the text data by converting it to lowercase and removing special characters.
- Then, I converted the text into numerical features using TF-IDF vectorization.
- After that, I trained a Linear SVM model on the processed data to classify messages as spam or not spam.
- The model achieved around 98% accuracy.
- Finally, I saved the trained model and vectorizer and integrated them into a Streamlit app for real-time prediction.

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py


