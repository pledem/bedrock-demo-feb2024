# RAG using LangChain with Amazon Bedrock Titan text, and embedding, using OpenSearch vector engine

## Prerequisites

1. This was tested on Python 3.11.4
2. It is advise to work on a clean environment, use `virtualenv` or any other virtual environment manager.

    ```bash
    pip install virtualenv
    python -m virtualenv venv
    source ./venv/bin/activate
    ```

3. Install requirements `pip install -r requirements.txt`

## Steps for using this sample code


1. We can start querying our LLM model Titan text in Amazon Bedrock with RAG

    ```bash
    python ask-bedrock.py --ask "your question here"
    ```

    >>Optional arguments:
    >>- `--index` to use a different index than the default **rag**
    >>- `--region` in case you are not using the default **us-east-1**
    >>- `--bedrock-model-id` to choose different models than Anthropic's Claude v2

## License

This library is licensed under the MIT-0 License.
