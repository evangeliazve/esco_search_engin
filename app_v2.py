import streamlit as st
import asyncio
from transformers import AutoTokenizer, AutoModelWithLMHead

import streamlit as st
import faiss
import torch
from transformers import BertTokenizer, BertModel

# Load the BERT model and tokenizer
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define your predefined list of possible results
results = [
    'apple',
    'banana',
    'orange',
    'grapes',
    'strawberry',
    'kiwi',
    'mango',
    'pear',
    'watermelon'
]

# Tokenize and encode the results using BERT
encoded_results = []
for result in results:
    input_ids = torch.tensor([tokenizer.encode(result, add_special_tokens=True)])
    with torch.no_grad():
        encoded_results.append(model(input_ids)[0][0][0].numpy())

# Create a Faiss index from the encoded results
index = faiss.IndexFlatL2(768)
index.add(encoded_results)

# Create a text input field in the Streamlit app
query = st.text_input('Enter a search query:')

# Use Faiss to retrieve the most similar results to the query
@st.cache
def get_suggestions(query):
    input_ids = torch.tensor([tokenizer.encode(query, add_special_tokens=True)])
    with torch.no_grad():
        encoded_query = model(input_ids)[0][0][0].numpy()
    distances, indices = index.search(encoded_query.reshape(1, -1), k=5)
    return [results[i] for i in indices[0]]

if len(query) > 1:
    suggestions = get_suggestions(query)

    # Display the list of suggestions to the user
    st.write('Suggestions:')
    for suggestion in suggestions:
        st.write(suggestion)
