import streamlit as st
from transformers import pipeline

# Load the zero-shot classification pipeline with BART model
classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

def classify_text(email):
    """
    Use the Facebook's Bart model to check if the email is spam or not

    Args:
        email(str): The email to classify
    Returns:
        str: The classification of the email
    """
    labels = ['spam', 'not spam']
    hypothesis_template = 'This is a {} email.'

    results = classifier(email, labels, hypothesis_template)
    return results['labels'][0]

# Streamlit app
st.title("Email Spam Classifier")
st.write("Enter an email below to classify it as 'spam' or 'not spam'.")

email_input = st.text_area("Email content", height=200)

if st.button("Classify"):
    if email_input:
        prediction = classify_text(email_input)
        st.write(f"The email is classified as: **{prediction}**")
    else:
        st.write("Please enter the email content to classify."