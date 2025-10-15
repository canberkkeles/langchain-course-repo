import os
import boto3
import json
from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import PromptTemplate
from opensearchpy import RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_community.embeddings import SagemakerEndpointEmbeddings
from langchain_community.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
from typing import Any, Dict, List


class ContentHandler(EmbeddingsContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, inputs: list[str], model_kwargs: Dict) -> bytes:
        """
        Transforms the input into bytes that can be consumed by SageMaker endpoint.
        Args:
            inputs: List of input strings.
            model_kwargs: Additional keyword arguments to be passed to the endpoint.
        Returns:
            The transformed bytes input.
        """
        # Example: inference.py expects a JSON string with a "inputs" key:
        input_str = json.dumps({"inputs": inputs, **model_kwargs})
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> List[List[float]]:
        """
        Transforms the bytes output from the endpoint into a list of embeddings.
        Args:
            output: The bytes output from SageMaker endpoint.
        Returns:
            The transformed output - list of embeddings
        Note:
            The length of the outer list is the number of input strings.
            The length of the inner lists is the embedding dimension.
        """
        # Example: inference.py returns a JSON string with the list of
        # embeddings in a "vectors" key:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json



load_dotenv()

INDEX_NAME=os.environ["INDEX_NAME"]
service = 'aoss'
region = 'us-east-1'
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key,
                   region, service, session_token=credentials.token)

def run_llm(query: str):
    content_handler = ContentHandler()
    embeddings = SagemakerEndpointEmbeddings(
        endpoint_name=os.environ["SM_ENDPOINT_NAME"],
        region_name="us-east-1",
        content_handler=content_handler,
    )
    docsearch = OpenSearchVectorSearch(index_name=INDEX_NAME,
                                opensearch_url=os.environ["OPENSEARCH_URL"],
                                embedding_function=embeddings,
                                http_auth=awsauth,
                                timeout=300,
                                use_ssl=True,
                                verify_certs=True,
                                connection_class=RequestsHttpConnection
                                )
    chat = ChatBedrockConverse(model="us.anthropic.claude-3-7-sonnet-20250219-v1:0")
    retrieval_qa_chat_prompt: PromptTemplate = hub.pull(
        "langchain-ai/retrieval-qa-chat",
    )
    stuff_documents_chain = create_stuff_documents_chain(
        chat, retrieval_qa_chat_prompt
    )
    qa = create_retrieval_chain(
        retriever=docsearch.as_retriever(), combine_docs_chain=stuff_documents_chain
    )
    result = qa.invoke(input={"input": query})
    return result
if __name__ == "__main__":
    res = run_llm(query="What is langchain?")

    print(res["answer"])