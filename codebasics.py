# -*- coding: utf-8 -*-

"""
Created on Tue Jul 16 14:00:37 2024

@author: USER
"""

import os
import streamlit as st
from langchain.llms import GooglePalm, OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
import google.generativeai as palm

google_api_key = "YOUR GOOGLE API KEY"
# Configure Google PaLM API
palm.configure(api_key= google_api_key)

# Initialize Google PaLM LLM
llm = GooglePalm(temperature=0.7, google_api_key=google_api_key)
#llm = OpenAI()
# Set up the Streamlit app
st.title("FAQS Answering Tool")
st.sidebar.title("Files")
main_holder = st.empty()
result = {"query": " ", "result": " "}

# Sidebar for file upload and query input
with st.sidebar:
    loader = CSVLoader("codebasics_faqs.csv", source_column="prompt")
    file = loader.load()
    query = st.text_input("Question: ")
    process_url_clicked = st.button("ASK")
    file_path_faiss = "faiss_index_codebasics"

    if process_url_clicked:
        docs = file
        embeddings = HuggingFaceInstructEmbeddings()

        if os.path.exists(file_path_faiss):
            # Load the FAISS index
            vectorIndex = FAISS.load_local(file_path_faiss, embeddings)
        else:
            vectorIndex = FAISS.from_documents(docs, embeddings)
            # Save the FAISS index
            vectorIndex.save_local(file_path_faiss)

        # Embed the query
        embeddings.embed_documents([query])

        # Create the retrieval chain
        chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorIndex.as_retriever())
        result = chain(query)

        st.success("File processed successfully!")

# Display the query and result
st.subheader("Question: ")
st.text(result['query'])
st.subheader("Answer: ")
st.text(result['result'])
