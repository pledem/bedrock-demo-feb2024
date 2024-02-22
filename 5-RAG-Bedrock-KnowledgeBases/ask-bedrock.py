import coloredlogs
import logging
import argparse
from langchain_community.embeddings import BedrockEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.retrievers import AmazonKnowledgeBasesRetriever
from langchain.llms.bedrock import Bedrock
import boto3


coloredlogs.install(fmt='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S', level='INFO')
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ask", type=str, default="What is <3?")
    parser.add_argument("--index", type=str, default="rag")
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--bedrock-model-id", type=str, default="anthropic.claude-v2")
    parser.add_argument("--bedrock-embedding-model-id", type=str, default="amazon.titan-embed-text-v1")
    
    return parser.parse_known_args()


def get_bedrock_client(region):
    bedrock_client = boto3.client("bedrock-runtime", region_name=region)
    return bedrock_client
    

def create_langchain_vector_embedding_using_bedrock(bedrock_client, bedrock_embedding_model_id):
    bedrock_embeddings_client = BedrockEmbeddings(
        client=bedrock_client,
        model_id=bedrock_embedding_model_id)
    return bedrock_embeddings_client
    


def create_bedrock_llm(bedrock_client, model_version_id):

    bedrock_llm = Bedrock(
        model_id=model_version_id, 
        client=bedrock_client,
        model_kwargs={'temperature': 0, "top_k": 10, "max_tokens_to_sample": 3000}
        )
    return bedrock_llm
    

def main():
    logging.info("Starting")
    args, _ = parse_args()
    question = args.ask
    region = args.region
    bedrock_model_id = args.bedrock_model_id
    
    # Creating all clients for chain
    bedrock_client = get_bedrock_client(region)
        
    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id="YNRV4FTNEE",
        retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 4}},
    )

    query = question

    retriever.get_relevant_documents(query=query)

    bedrock_llm = create_bedrock_llm(bedrock_client, bedrock_model_id)

    qa = RetrievalQA.from_chain_type(
        llm=bedrock_llm, retriever=retriever, return_source_documents=True
    )

    response = qa(question)
    
    logging.info(f"The answer from Bedrock {bedrock_model_id} is: {response.get('result')}")
    logging.info(f"Source Attribution : {response.get('source_documents')}")
    

if __name__ == "__main__":
    main()