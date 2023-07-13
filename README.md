# Open-Source LLM and Open-Source Embedding Model
This project contains different tests with open-source models. The results are not that resilient, but they are a good starting point for further research.
- searchWithLLM.py: We tried to replicate the langchain RetrievalQAWithSourcesChain (Flan-T5s Token limit was not high enough to use that chain). Therefore, there are different prompt templates. The results are not that good, but maybe you find a prompt, that works well?
- chatWithLLM.py: You can basically write a prompt to flan-t5, and it will answer you. 

## Steps to run the cloud function locally:
1. In your terminal run: ```python3 -m venv env``` 
2. In your terminal run: ```source env/bin/activate```
3. In your terminal run: ```python3 -m pip install -r requirements.txt```
4. Run the createDatabase.py file to create the database: ```python3 createDatabase.py```
5. Run the searchWithLLM file: ```python3 searchWithLLM.py```
6. Or run the chatWithLLM file: ```python3 chatWithLLM.py```

## Further information
- To run the llms locally, you need about 6 - 10 gb ram
- If you need further help, you can contact us at: chrissy.drx@gmail.com
