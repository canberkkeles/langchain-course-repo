import os
import boto3
from opensearchpy import RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_aws import BedrockEmbeddings,ChatBedrockConverse
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.prompts import PromptTemplate
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()

def main():
    service = "aoss"  # must set the service as 'aoss'
    region = "us-east-1"
    credentials = boto3.Session().get_credentials()
    awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)
    
    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
    llm = ChatBedrockConverse(model="us.anthropic.claude-3-5-haiku-20241022-v1:0")
    
    query = "what is Pinecone in machine learning?"
    chain = PromptTemplate.from_template(template=query) | llm
    
    
    vectorstore = OpenSearchVectorSearch(
        embedding_function=embeddings,
        opensearch_url=os.environ["OPENSEARCH_URL"],
        http_auth=awsauth,
        timeout=300,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        index_name=os.environ["INDEX_NAME"],
    )
    
    retrieval_qa_chat_prompt=hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrival_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )
    
    result = retrival_chain.invoke(input={"input": query})
    print(result)
    
if __name__ == "__main__":
    main()