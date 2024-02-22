import coloredlogs
import logging
import argparse
from langchain.prompts import PromptTemplate
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
    

def create_bedrock_llm(bedrock_client, model_version_id):
    bedrock_llm = Bedrock(
        model_id=model_version_id, 
        client=bedrock_client,
        model_kwargs={'temperature': 0}
        )
    return bedrock_llm
    

def main():
    logging.info("Starting")
    args, _ = parse_args()
    ask=args.ask
    region = args.region
    bedrock_model_id = args.bedrock_model_id
    
    # Creating all clients for chain
    bedrock_client = get_bedrock_client(region)
    bedrock_llm = create_bedrock_llm(bedrock_client, bedrock_model_id)
    
    # LangChain prompt template
    if len(args.ask) > 0:
        question = args.ask
    else:
        question = "what is the meaning of <3?"
        logging.info(f"No question provided, using default question {question}")
    
    
    
    # create the prompt
    prompt_template: str = """/
        You are a doctor, give responses to the following/ 
        question: {question}. Do not use technical words, give easy/
        to understand responses.
    """
    
    prompt = PromptTemplate.from_template(template=prompt_template)

    # format the prompt to add variable values
    prompt_formatted_str: str = prompt.format(
        question=ask)
    
    logging.info(prompt_formatted_str) 

    response = bedrock_llm.invoke(prompt_formatted_str)
        
    logging.info(f"The answer from Bedrock {bedrock_model_id} is: {response}")
    

if __name__ == "__main__":
    main()