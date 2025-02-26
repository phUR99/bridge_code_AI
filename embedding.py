from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
import dotenv
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.document_loaders import PyPDFLoader, DataFrameLoader
import pandas as pd


def embedding(vectors, query: list, data_path:str='data.csv'):
    
    docs = vectors
    ret =[]
    for elements in query:
        retrieved_docs = db.similarity_search(query=elements, k=3)
        for context in retrieved_docs: 
            ret.append(context.to_json())
    return ret

def vectorspace(model, data_path:str="data.csv"):
    df = pd.read_csv(filepath_or_buffer= data_path)
    loader = DataFrameLoader(df, page_content_column="output")
    docs = loader.load()
    db = Chroma.from_documents(documents=docs, embedding=model)
    return db
    

if __name__ == '__main__':

    dotenv.load_dotenv()
    model_name = "BAAI/bge-small-en"
    model_kwargs = {'device':'cpu'}
    encode_kwargs = {'normalize_embeddings' : True}
    hf = HuggingFaceEmbeddings(
        model_name = model_name,
        model_kwargs = model_kwargs,
        encode_kwargs = encode_kwargs
    )
    db = vectorspace(hf)
    import time
    start = time.time()
    print(embedding(db, ["for"]))
    print(embedding(db, ["for"]))
    print(embedding(db, ["for"]))
    print(time.time() - start)
    
    