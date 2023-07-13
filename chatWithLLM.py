from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
import os
from langchain.embeddings import HuggingFaceEmbeddings


def chat_llm():
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

    userinput = input("What do you want to say to the LLM? ")
    print(flan_llm.predict(userinput))
    return


if __name__ == '__main__':
    chat_llm()
