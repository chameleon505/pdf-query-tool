import argparse
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
import pinecone


def load_and_process_pdf(pdf_name: str, query: str, huggingfacehub_api_token: str, pinecone_api_key: str,
                         index_name: str, namespace: str = "default"):
    # Load the PDF
    loader = PyPDFLoader(pdf_name)
    data = loader.load()

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = text_splitter.split_documents(data)

    # Set up embeddings
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    # Initialize Pinecone
    pinecone.init(api_key=pinecone_api_key)
    index = pinecone.Index(index_name)

    # Create vector store
    vectorstore = PineconeVectorStore(
        index=index,
        pinecone_api_key=pinecone_api_key,
        embedding=embeddings,
        namespace=namespace,
        index_name=index_name
    )

    # Add documents to the vector store
    vectorstore.add_texts(texts=[t.page_content for t in docs])

    # Perform similarity search
    result_docs = vectorstore.similarity_search(query, k=1)

    return result_docs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a PDF and perform a similarity search.")
    parser.add_argument("pdf_name", type=str, help="The name of the PDF file to process.")

    args = parser.parse_args()
    pdf_name = args.pdf_name

    query = "Transport Layer Security (TLS)"
    HUGGINGFACEHUB_API_TOKEN = "your_huggingface_api_token"
    PINECONE_API_KEY = "your_pinecone_api_key"
    index_name = "your_index_name"

    # Call the function with command-line arguments
    docs = load_and_process_pdf(pdf_name, query, HUGGINGFACEHUB_API_TOKEN, PINECONE_API_KEY, index_name)
    print(docs)
