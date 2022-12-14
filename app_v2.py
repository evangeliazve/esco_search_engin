import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Use the AutoTokenizer and AutoModelForQuestionAnswering classes
# to load the BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")

# Create a text input field using Streamlit
text_input = st.text_input("Enter a query:")

# Use the BERT tokenizer to tokenize the input text
tokens = tokenizer.tokenize(text_input)

# Use the BERT model to predict the most likely completion for the input text
prediction = model.predict(tokens)

# Use the `search` method to find indexed results that match the predicted completion
results = model.search(prediction)

# Display the search results in a Streamlit output field
st.write("Search results:", results)
