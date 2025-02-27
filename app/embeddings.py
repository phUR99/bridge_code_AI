from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
import dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, DataFrameLoader
import pandas as pd


def embedding(vectors, query: list, data_path:str='data.csv'):
    
    docs = vectors
    ret =[]
    retrieve =  docs.as_retriever(search_type="mmr", search_kwargs={'k':3, 'fetch_k':10})
    for elements in query:
        retrieved_docs = retrieve.invoke(input=elements)
        for context in retrieved_docs: 
            ret.append(context.page_content)
    return ret

def vectorspace(model, data_path:str="data.csv"):
    df = pd.read_csv(filepath_or_buffer= data_path)
    loader = DataFrameLoader(df, page_content_column="output")
    docs = loader.load()
    db = Chroma.from_documents(documents=docs, embedding=model, persist_directory="./chorma_db")
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
    import os
    if os.path.exists("chorma_db"):
        print("load")
        db = Chroma(persist_directory="./chorma_db", embedding_function=hf)
    else:
        print("make")
        db = vectorspace(model=hf)

    import time
    start = time.time()
    result = embedding(db, ['elif'])
    for ret in result:
        print(result[0])

    
    
