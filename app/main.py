# app/main.py
from langchain_huggingface import  HuggingFaceEmbeddings
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from langchain_chroma import Chroma
import utils
# LLM 클래스 가져오기
from llm import LLM
from embeddings import embedding, vectorspace

# 환경 변수 로드
load_dotenv()

model_name = "BAAI/bge-small-en"
model_kwargs = {'device':'cpu'}
encode_kwargs = {'normalize_embeddings' : True}
hf = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en", model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
print("model!")
if os.path.exists("chorma_db"):
    print("load")
    db = Chroma(persist_directory="./chorma_db", embedding_function=hf)
else:
    print("make")
    db = vectorspace(model=hf)

# FastAPI 앱 초기화
app = FastAPI()

# API 보안을 위한 HTTPBearer
security = HTTPBearer()

# API 토큰 (Java 백엔드와 통신 시 검증)
API_TOKEN = os.getenv("API_TOKEN")

# LLM 클래스 초기화
llm = None


#  요청 데이터 모델 정의 (Pydantic 사용)
class InitialQueryRequest(BaseModel): # 이 변수들의 이름을 백엔드 서버에 알려줘서 json으로 보낼때 맞추기 그래야 인식되어 값을 가져올수 있음
    user_id: str
    original_codes: str
class UserInputRequest(BaseModel):
    user_id: str
    oneline_code : str
class FinalSummaryRequest(BaseModel):
    user_id: str


#  FastAPI 엔드포인트: 사용자 입력을 받아 임베딩, 검색, LLM 호출
@app.post("/interpret_initial_code/")
def interpret_initial_code_api(request: InitialQueryRequest):
    """
    사용자 입력을 임베딩 -> 벡터 DB에서 검색 -> LLM에 전달 -> 응답 반환
    """
    global db, llm
    # API 토큰 검증
    
    input_list = utils.str_to_list(request.original_codes)
    #  1. 유저 입력을 임베딩
    embedding_text = embedding(vectors=db, query=input_list) # 여러줄로 나눠서 받아야 할듯
    #  2. 벡터 데이터베이스에서 검색
    #context_docs = query_vectorstore(embedding, n_results=3)
    #print(type(embedding_text[0]))
    #  3. LLM 호출 및 응답 반환
    llm = LLM(request.original_codes)
    response = llm.interpret_initial_code(
        user_id=request.user_id,
        original_codes = request.original_codes,
        context_docs=embedding_text,
        prompt_model= "initial_model"
    )
    #print(response)
    print(llm.original_codes)
    return {"response":response}

@app.post("/interpret_user_input/")
def interpret_user_input_api(request: UserInputRequest):
    
    # API 토큰 검증
    global db, llm
    print(llm.original_codes)
    try:
        input_list = utils.str_to_list(request.oneline_code)
        #  1. 유저 입력을 임베딩
        embedding_text = embedding(vectors=db, query=input_list)
        #  2. 벡터 데이터베이스에서 검색
        #context_docs = query_vectorstore(embedding, n_results=3)
        #print(input_list)
        #print(request.oneline_code)
        #  3. LLM 호출 및 응답 반환
        response = llm.interpret_initial_code(
            user_id=request.user_id,
            oneline_code = request.oneline_code,
            context_docs = embedding_text,
            prompt_model = "user_input_model"
            
        )
        return {"response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize-user-inputs/")
def summarize_user_inputs_api(request: FinalSummaryRequest):

    # API 토큰 검증
    global llm
    try:
        response = llm.interpret_initial_code(
            user_id = request.user_id,
            prompt_model = "summarize_model"
        )
        return {"response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 서버 상태 확인 (테스트용)
@app.get("/health/")
def health_check():
    """서버 상태 확인을 위한 엔드포인트"""
    return {"status": "AI server is running"}
