#1 necessary imports
import coloredlogs
import logging
import argparse

from langchain.llms.bedrock import Bedrock

coloredlogs.install(fmt='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S', level='INFO')
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ask", type=str, default="What is <3?")
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--bedrock-model-id", type=str, default="anthropic.claude-v2")
    
    return parser.parse_known_args()


#2 Write a function for invoking model- client connection with Bedrock with profile, model_id & Inference params- model_kwargs
def create_bedrock_llm(bedrock_model_id):
    bedrock_llm = Bedrock(
       credentials_profile_name='default',
       model_id=bedrock_model_id,
       model_kwargs= {
        "temperature": 0.5,
        "top_p": 1,
        "top_k": 250,
        "max_tokens_to_sample": 500})
    return bedrock_llm


def main():
    logging.info("Starting")
    args, _ = parse_args()
    ask=args.ask
    bedrock_model_id = args.bedrock_model_id
    
    #3 Test out the LLM with Invoke method
    bedrock_llm = create_bedrock_llm(bedrock_model_id)

    response = bedrock_llm.invoke(ask)
    
    logging.info(f"The answer from Bedrock {bedrock_model_id} is: {response}")

   
if __name__ == "__main__":
    main()