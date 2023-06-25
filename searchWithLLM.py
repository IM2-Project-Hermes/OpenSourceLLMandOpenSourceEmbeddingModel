import os
from langchain.llms import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
import time
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.chains import RetrievalQAWithSourcesChain
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
import os
from langchain.vectorstores import Chroma
from chromadb.utils import embedding_functions
from langchain.embeddings import HuggingFaceEmbeddings


def ask_llm(question):
    load_dotenv()

    # Initialize the models
    model_id = 'google/flan-t5-large'

    # Save the models, if they don't exist
    if not os.path.isdir('./models/flan-t5-large'):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        model.save_pretrained('./models/flan-t5-large')

    if not os.path.isdir('./models/flan-t5-large-tokenizer'):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained('./models/flan-t5-large-tokenizer')

    model = AutoModelForSeq2SeqLM.from_pretrained('./models/flan-t5-large')
    tokenizer = AutoTokenizer.from_pretrained('./models/flan-t5-large-tokenizer')

    # Create the pipeline
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=200
    )

    flan_llm = HuggingFacePipeline(pipeline=pipe)

    # all-MiniLM-L6-v2 embedding function from HuggingFace
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # Initialize the database
    chroma_client = chromadb.Client(
        Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="db",
        )
    )

    collection = chroma_client.get_collection(name="documents")


    template = """
    Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES").
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    ALWAYS return a "SOURCES" part in your answer.
    Answer the question: {question} based on this data: {summaries}
    """

    template = """
    {question}
    summarize: {summaries}
    """

    documents = collection.query(
        query_texts=[question],
        n_results=2
    )

    print(documents)

    # Different example templates for the LLM

    template = f"""
summarize based on the question {question}: 
- {documents['documents'][0][0]} 
- {documents['documents'][0][1]}
"""

    template = f"""
Given the following extracted parts of a long document and a question, create a final answer with references.
Answer the question: {question} based on this data:
- {documents['documents'][0][0]}
- {documents['documents'][0][1]}
Rephrase that answer in your own words
"""

    template = f"""
Given the following extracted parts of a long document and a question, create a final answer with references.
Answer the question: {question} based on this data:
- {documents['documents'][0][0]}
- {documents['documents'][0][1]}
Rephrase that answer in your own words
    """

    template = f"""
summarize: {documents['documents'][0][0]} {documents['documents'][0][1]}
    """

    print(template)
    print("Answer: " + flan_llm.predict(template))
    return


if __name__ == '__main__':
    ask_llm("How does the onboarding process work?")
