#1. Necessary Imports 
import coloredlogs
import logging
import argparse

import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms.bedrock import Bedrock

coloredlogs.install(fmt='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S', level='INFO')
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ask", type=str, default="What is <3?")
    parser.add_argument("--bedrock-model-id", type=str, default="anthropic.claude-v2")
    parser.add_argument("--bedrock-embedding-model-id", type=str, default="amazon.titan-embed-text-v1")
    
    return parser.parse_known_args()


def main():

    args, _ = parse_args()
    ask = args.ask
    
    index = create_index()

    response_text=rag_response(index, question=ask) 
    logging.info(f"The answer from Bedrock is: {response_text}")


def create_index():
    
    #2. Define the data source and load data with PDFLoader
    #data_load=PyPDFLoader('https://docs.aws.amazon.com/pdfs/whitepapers/latest/aws-serverless-data-analytics-pipeline/aws-serverless-data-analytics-pipeline.pdf')
    data_load=PyPDFLoader('https://www.accessdata.fda.gov/drugsatfda_docs/label/2009/019962s038lbl.pdf')
    
    #3. Split the Text based on Character, Tokens etc. - Recursively split by character - ["\n\n", "\n", " ", ""]
    data_split=RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=100,chunk_overlap=10)

    #4. Create Embeddings -- Client connection
    data_embeddings=BedrockEmbeddings(
        credentials_profile_name= 'default',
        model_id='amazon.titan-embed-text-v1')
    
    #5 Create Vector DB, Store Embeddings and Index for Search - VectorstoreIndexCreator
    data_index=VectorstoreIndexCreator(
        text_splitter=data_split,
        embedding=data_embeddings,
        vectorstore_cls=FAISS)
    
    #6  Create index using the PDF Document
    db_index=data_index.from_loaders([data_load])
    return db_index


#6a. Write a function to connect to Bedrock Foundation Model - Claude Foundation Model
def invoke_llm():
    llm=Bedrock(
        credentials_profile_name='default',
        model_id='anthropic.claude-v2',
        model_kwargs={
        "max_tokens_to_sample":3000,
        "temperature": 0.1,
        "top_p": 0.9})
    return llm


#6b. Write a function which searches the user prompt, searches the best match from Vector DB and sends both to LLM.
def rag_response(index,question):
    rag_llm=invoke_llm()
    hr_rag_query=index.query(question=question,llm=rag_llm)
    return hr_rag_query



if __name__ == "__main__":
    main()