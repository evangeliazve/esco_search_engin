import streamlit as st
import pickle
import altair as alt
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import numpy as np

import pandas as pd
from transformers import AutoModelWithLMHead, AutoTokenizer
from scipy.spatial import distance

import transformers

# Load the BERT model
model = transformers.BertModel.from_pretrained('bert-base-cased')
suggestion_list = ["apple", "banana","vegatable", "data scientist", "data analyst"]
def get_autocompletion_suggestions(input_text, suggestion_list, top_k=5):
    input_ids = transformers.BertTokenizer.from_pretrained('bert-base-cased').encode(input_text, return_tensors='pt')
    output = model(input_ids)[0]
    # Use Faiss to find the top k semantically similar suggestions from the suggestion list
    index = faiss.IndexFlatL2(output.shape[1])
    index.add(output)
    distances, indices = index.search(output, top_k)
    return [suggestion_list[i] for i in indices[0]]

input_text = st.text_input("Enter your text:")
if input_text:
    suggestions = get_autocompletion_suggestions(input_text, suggestion_list)
    st.write(f'Autocompletion suggestions: {suggestions}')
