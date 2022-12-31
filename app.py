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

model = AutoModelWithLMHead.from_pretrained("bert-base-cased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

st.title("Autosuggestion App")

data = ["apple", "banana", "orange", "grape", "strawberry"]

input_text = st.text_input("Enter at least 2 characters")
suggestions = []

if len(input_text) > 1:
  input_tokens = tokenizer.encode(input_text, return_tensors="pt").to("cpu")
  input_vec = model(input_tokens)[0][0][0]
  suggestions = []
  for item in data:
    item_tokens = tokenizer.encode(item, return_tensors="pt").to("cpu")
    item_vec = model(item_tokens)[0][0][0]
    sim = distance.cosine(input_vec, item_vec)
    if sim < 0.5:
      suggestions.append(item)
  suggestions = sorted(suggestions, key=lambda x: distance.cosine(input_tokens, tokenizer.encode(x)))

st.write("Suggestions:", suggestions)

custom_text = st.text_input("Enter your own text")
if custom_text != "":
  st.write("You entered:", custom_text)

st.run()
