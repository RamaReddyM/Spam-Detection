Spam Detection Web App using Streamlit and Naive Bayes Classifier
This repository contains code for a web application built with Streamlit that predicts whether a given message is spam or not, using a Naive Bayes classifier trained on SMS data.

Project Structure
spam.csv: Dataset containing SMS messages labeled as spam or not spam.
spamdetection.py: Main Python script that loads the dataset, trains a model, and deploys the Streamlit web app.
README.md: This file, providing an overview of the project, setup instructions, and usage details.
Dependencies
Ensure you have the following Python libraries installed:

pandas
scikit-learn (for CountVectorizer and MultinomialNB)
streamlit
You can install them using pip:

Copy code
pip install pandas scikit-learn streamlit
Usage
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/spam-detection.git
cd spam-detection
Run the Streamlit app:

arduino
Copy code
streamlit run app.py
Open the Streamlit app:

After running the above command, a new tab will open in your default web browser displaying the Spam Detection web app.
Enter a message into the input box and click the "Validate" button to see if it's classified as spam or not.
Explanation of app.py
Imports and Libraries
python
Copy code
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
Imports necessary libraries:
pandas for data handling.
sklearn modules for machine learning tasks (train_test_split, CountVectorizer, MultinomialNB).
streamlit for creating and deploying web applications.
Data Loading and Preprocessing
python
Copy code
data = pd.read_csv("spam.csv")
data.drop_duplicates(inplace=True)
data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])
Loads SMS data from spam.csv, removes duplicate entries, and converts categories from 'ham' and 'spam' to 'Not Spam' and 'Spam'.
Model Training
python
Copy code
msg = data['Message']
cat = data['Category']
(msg_train, msg_test, cat_train, cat_test) = train_test_split(msg, cat, test_size=0.2)
cv = CountVectorizer(stop_words='english')
features = cv.fit_transform(msg_train)
model = MultinomialNB()
model.fit(features, cat_train)
Splits the data into training and testing sets.
Uses CountVectorizer to convert text messages into numerical feature vectors.
Trains a Multinomial Naive Bayes classifier on the training data.
Streamlit Web Application
python
Copy code
st.set_page_config(page_title="Spam Detection", page_icon="📧")
st.title("Spam Detection 📧")
st.header("Enter a message to check if it's spam or not:")
input_msg = st.text_input('Type your message here...')
if st.button('Validate'):
    output = predict(input_msg)
    if output[0] == 'Not Spam':
        st.success("This message is **Not Spam**!", icon="✅")
    else:
        st.error("This message is **Spam**!", icon="❌")
Configures Streamlit page settings and displays a title and header.
Provides a text input box for users to enter a message.
Implements a button that triggers a prediction function (predict) when clicked.
Shows a success or error message based on the model's prediction.
Prediction Function
python
Copy code
def predict(message):
    predictinput = cv.transform([message]).toarray()
    result = model.predict(predictinput)
    return result
Defines a function (predict) that takes a message as input, transforms it into a numerical vector, predicts its category ('Spam' or 'Not Spam'), and returns the result.
This README provides a structured overview of the project, including setup instructions, dependencies, usage guidelines, and an explanation of the key components (spamdetection.py). It helps users understand how to run the Spam Detection web app and what to expect when using it.
