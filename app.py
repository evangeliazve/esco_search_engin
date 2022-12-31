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
  output = model.generate(input_tokens, max_length=20, top_p=0.9, top_k=10, temperature=0.8)
  output_text = tokenizer.decode(output[0], skip_special_tokens=True)
  suggestions = output_text.split(" ")
  suggestions = [s for s in suggestions if s in data]
  suggestions = sorted(suggestions, key=lambda x: distance.cosine(input_tokens, tokenizer.encode(x)))

st.write("Suggestions:", suggestions)

custom_text = st.text_input("Enter your own text")
if custom_text != "":
  st.write("You entered:", custom_text)
