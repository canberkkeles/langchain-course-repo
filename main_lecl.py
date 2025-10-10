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
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

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
    
    prompt = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three senteces at maximum and keep the answer as concise as possible.
    Always say "Thank you for asking!" at the end of the answer.
    
    {context}
    
    Question: {question}
    Helpful Answer:"""
    
    custom_rag_prompt=PromptTemplate.from_template(prompt)
    
    rag_chain = (
        {"context": vectorstore.as_retriever() | format_docs,"question": RunnablePassthrough()}
        | custom_rag_prompt 
        | llm
        )
    
    result = rag_chain.invoke(query)
    print(result)
    
if __name__ == "__main__":
    main()