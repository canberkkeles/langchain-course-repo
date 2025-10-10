import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_aws import BedrockEmbeddings,ChatBedrockConverse
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub

load_dotenv()

def main():
    print("Hello world!")
    FILE_PATH=""
    loader = PyPDFLoader(FILE_PATH)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents)
    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
    llm = ChatBedrockConverse(model="us.anthropic.claude-3-5-haiku-20241022-v1:0")
    vector_store = FAISS.from_documents(docs,embeddings)
    vector_store.save_local("faiss-index-react")
    new_vector_store = FAISS.load_local("faiss-index-react",embeddings,
        allow_dangerous_deserialization=True)
    
    retrieval_qa_chat_prompt=hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrival_chain = create_retrieval_chain(
        retriever=vector_store.as_retriever(), combine_docs_chain=combine_docs_chain
    )
    
    query="Give me the gist of ReAct in 3 sentences"
    result = retrival_chain.invoke(input={"input": query})
    print(result)
    

if __name__ == "__main__":
    main()