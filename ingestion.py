import asyncio
import os
import ssl
from typing import Any, Dict, List
import certifi
import boto3
from opensearchpy import RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import (Colors, log_error, log_header, log_info, log_success,
                    log_warning)
import json
from typing import Dict, List

from langchain_community.embeddings import SagemakerEndpointEmbeddings
from langchain_community.embeddings.sagemaker_endpoint import EmbeddingsContentHandler


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

ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

service = 'aoss'
region = 'us-east-1'
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key,
                   region, service, session_token=credentials.token)

content_handler = ContentHandler()
embeddings = SagemakerEndpointEmbeddings(
    endpoint_name=os.environ["SM_ENDPOINT_NAME"],
    region_name="us-east-1",
    content_handler=content_handler,
)

vectorstore = OpenSearchVectorSearch(index_name=os.environ["INDEX_NAME"],
                                opensearch_url=os.environ["OPENSEARCH_URL"],
                                embedding_function=embeddings,
                                http_auth=awsauth,
                                timeout=300,
                                use_ssl=True,
                                verify_certs=True,
                                connection_class=RequestsHttpConnection
                                )


tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


def index_documents_async(documents: List[Document], batch_size: int = 50):
    """Process documents in batches asynchronously."""
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"📚 VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN,
    )
    
    # Create batches
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"📦 VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )
    
    # Process all batches concurrently
    def add_batch(batch: List[Document], batch_num: int):
        try:
            vectorstore.add_documents(batch)
            log_success(
                f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
            )
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True
        
    # Process batches concurrently
    results = []
    for i, batch in enumerate(batches):
        res = add_batch(batch, i + 1)
        results.append(res)

    # Count successful batches
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully"
        )

def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "🗺️  TavilyCrawl: Starting to crawl the documentation site",
        Colors.PURPLE,
    )
    
    res = tavily_crawl.invoke({
        "url": "https://python.langchain.com/",
        "max_depth": 2,
        "extract_depth": "advanced"
    })
    all_docs = res["results"]
    print(all_docs[0])
    docs_array = [Document(page_content=doc["raw_content"],
            metadata={"source": doc["url"]}) for doc in all_docs]
    # Split documents into chunks
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"✂️  Text Splitter: Processing {len(docs_array)} documents with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(docs_array)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(docs_array)} documents"
    )
    # Process documents asynchronously
    index_documents_async(splitted_docs, batch_size=32)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")
    
if __name__ == "__main__":
    main()