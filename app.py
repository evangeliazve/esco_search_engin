import streamlit as st
import pickle
import altair as alt
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import numpy as np

#@st.cache
def read_data():
    with open('fonctions_list.pickle', 'rb') as h:
        return pickle.load(h)

#@st.cache
#def read_author_data():
#    with open('author_data.pickle', 'rb') as h:
#        return pickle.load(h)
    
#@st.cache
#def unique_fos_level(df):
 #   return sorted(df['level'].unique())[1:]

#def unique_fos(df, level, num):
#    return list(df[df['level']==level].name.value_counts().index[:num])

@st.cache(allow_output_mutation=True)
def load_bert_model(name='all-MiniLM-L6-v2'):
    # Instantiate the sentence-level DistilBERT
    return SentenceTransformer(name)

@st.cache
def load_faiss_index():
    with open('faiss_index_esco.pickle', 'rb') as h:
        return pickle.load(h)
@st.cache
def vector_search(query, model, index, num_results=10):
    """Tranforms query to vector using a pretrained, sentence-level 
    DistilBERT model and finds similar vectors using FAISS.
    Args:
        query (str): User query that should be more than a sentence long.
        model (sentence_transformers.SentenceTransformer.SentenceTransformer)
        index (`numpy.ndarray`): FAISS index that needs to be deserialized.
        num_results (int): Number of results to return.
    Returns:
        D (:obj:`numpy.array` of `float`): Distance between results and query.
        I (:obj:`numpy.array` of `int`): Paper ID of the results.
    
    """
    vector = model.encode(list(query))
    D, I = index.search(np.array(vector).astype("float32"), k=5)
    return [j for i,j in enumerate(I[0])]

def vector_search_d(query, model, index, num_results=10):
    """Tranforms query to vector using a pretrained, sentence-level 
    DistilBERT model and finds similar vectors using FAISS.
    Args:
        query (str): User query that should be more than a sentence long.
        model (sentence_transformers.SentenceTransformer.SentenceTransformer)
        index (`numpy.ndarray`): FAISS index that needs to be deserialized.
        num_results (int): Number of results to return.
    Returns:
        D (:obj:`numpy.array` of `float`): Distance between results and query.
        I (:obj:`numpy.array` of `int`): Paper ID of the results.
    
    """
    vector = model.encode(list(query))
    D, I = index.search(np.array(vector).astype("float32"), k=5)
    return [j for i,j in enumerate(D[0])]

def main():
    data = read_data()
    model = load_bert_model()
    faiss_index = faiss.deserialize_index(load_faiss_index())
       
    st.title("Moteur de recherche fonctions")
        
    # User search
    user_input = st.text_input("Search by query")
    encoded_user_input = vector_search([user_input], model, faiss_index)
    encoded_user_input_d = vector_search_d([user_input], model, faiss_index)
    data = pd.DataFrame(data)
    data["id"] = data.index
    frame_1 = data[data.id.isin(encoded_user_input)] 
    frame_2 = data[data.id.isin(encoded_user_input_d)]
    frame = pd.merge(frame_1, frame_2, on = "id")
    st.write(frame_1)
             
if __name__ == '__main__':
    main()
