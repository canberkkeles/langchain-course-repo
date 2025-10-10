import os
import boto3
from opensearchpy import RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch

load_dotenv()

def main():

    service = "aoss"  # must set the service as 'aoss'
    region = "us-east-1"
    FILE_PATH = ""
    credentials = boto3.Session().get_credentials()
    awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)
    
    loader = TextLoader(FILE_PATH)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    embeddings_client = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
    vector_store = OpenSearchVectorSearch.from_documents(
        docs,
        embeddings_client,
        opensearch_url=os.environ["OPENSEARCH_URL"],
        http_auth=awsauth,
        timeout=300,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        index_name=os.environ["INDEX_NAME"],
        engine="faiss",
    )

if __name__ == "__main__":
    main()
