import pandas as pd
import numpy as np
import dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
dotenv.load_dotenv()
#data = pd.read_csv('data.csv')

def gen(ref:str):
    model = ChatOpenAI(model="gpt-4o-mini", temperature=1)

    # 프롬프트 템플릿 정의
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", 
            "당신은 Python 프로그래밍 퀴즈를 출제하는 교사입니다. "
            "사용자가 입력한 주제와 관련된 코딩 문제를 생성하세요. "
            "코드는 입문자가 쉽게 이해할 수 있도록 작성하며, "
            "변수명과 함수 이름은 직관적으로 설정해야 합니다. "
            "출력 결과는 예측 가능해야 한다."
            "문자는 반드시 작은따음표로 묶어줘."
            "코드는 6줄 이상 8줄 이내로 작성하고, 주석은 포함하지 마세요."
            "코드만 작성해주세요"
                    ),
            ("user", "{ref}")
        ]
    )
    formatted_prompt = prompt.format(ref=ref)
    result = model.invoke(formatted_prompt)
    return result.content


def topic_gen(ref:str):
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 프롬프트 템플릿 정의
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", 
                "당신은 Python 코드 해석 전문가입니다. "
                "사용자가 코드를 입력하면, 해당 코드가 해결하려는 원래 문제를 예측하세요. "
                "출제자의 시선에서 반드시 한 문장(20자 이하)으로 표현하세요. "
                "코드에 포함된 변수를 반드시 문장에 포함하며, "
                "항상 명령조로 문장을 끝내라."
            ),
            ("user", "{ref}")
        ]
    )
    formatted_prompt = prompt.format(ref=ref)
    result = model.invoke(formatted_prompt)
    return result.content
def comment(ref:str):
    model = ChatOpenAI(model="gpt-4o-mini", temperature=1)
    prompt = ChatPromptTemplate.from_messages(
            [
                ("system", 
                        "당신은 Python 프로그래밍을 가르치는 교육 도우미입니다. 사용자가 최종적으로 완성해야 할 파이썬 코드를 입력하면, "
                        "각 코드 줄마다 하나의 주석을 작성해야 합니다. "
                        "주석은 반드시 개행 문자로 끝난다."
                        "개행 문자는 반드시 문장당 한 번만 사용한다."
                        "주석은 코드가 수행해야 할 작업을 설명합니다."
                        "주석의 시작에는 항상 #이 들어갑니다."
                        "너무 복잡한 개념을 한꺼번에 설명하지 말고, 처음으로 배우는 사람이 이해하기 쉽게 단계적으로 안내하세요. "
                        "반복 횟수, 변수명, 조건 등은 명확하게 서술하세요"
                        "만약 코드에 빈 줄이 포함되어 있다면 무시하세요."
                        "반드시 주석만을 제시해줘"
                        ),
                ("user", "{ref}")
            ]
        )
    formatted_prompt = prompt.format(ref=ref)
    result = model.invoke(formatted_prompt)
    return result.content

python_basics_50 = [
    "변수 선언 및 할당",
    "데이터 타입 (int, float, str, bool)",
    "type() 함수로 데이터 타입 확인",
    "print() 함수 사용법",
    "input() 함수로 사용자 입력 받기",
    "기본 연산자 (+, -, *, /, //, %, **)",
    "문자열 포매팅 (f-string, format(), % 연산자)",
    "문자열 인덱싱 및 슬라이싱",
    "문자열 내장 메서드 (strip, replace, split, join 등)",
    "len() 함수로 문자열 길이 구하기",
    "리스트 기본 사용법",
    "리스트 인덱싱 및 슬라이싱",
    "리스트 추가 및 삭제 (append, insert, remove, pop)",
    "리스트 정렬 및 역순 (sort, reverse)",
    "리스트 내장 메서드 (index, count 등)",
    "튜플 기본 사용법",
    "튜플과 리스트의 차이점",
    "딕셔너리 기본 사용법",
    "딕셔너리 키-값 추가 및 삭제",
    "딕셔너리 내장 메서드 (keys, values, items, get)",
    "집합 (set) 기본 사용법",
    "집합 연산 (합집합, 교집합, 차집합)",
    "조건문 (if, elif, else)",
    "비교 연산자 (==, !=, >, <, >=, <=)",
    "논리 연산자 (and, or, not)",
    "반복문 (for, while)",
    "range() 함수 사용법",
    "enumerate() 함수 사용법",
    "break와 continue",
    "리스트 컴프리헨션",
    "함수 정의 및 호출 (def)",
    "함수의 기본 매개변수 및 반환값",
    "가변 인자 (*args, **kwargs)",
    "람다 함수 (lambda)",
    "예외 처리 (try, except, finally)",
    "파일 입출력 (open, read, write)",
    "with문을 활용한 파일 처리",
    "클래스 및 객체 지향 프로그래밍 기본 (class, object)",
    "생성자 및 소멸자 (__init__, __del__)",
    "클래스 변수와 인스턴스 변수",
    "상속 및 오버라이딩",
    "모듈 임포트 (import, from, as)",
    "표준 라이브러리 사용 (math, datetime, random 등)",
    "리스트, 딕셔너리, 튜플의 반복문 활용",
    "zip() 함수 활용",
    "map()과 filter() 함수 활용",
    "리스트 정렬 (sorted, key 매개변수 활용)",
    "이진 탐색 (bisect 모듈)",
    "json 모듈 활용 (json 파일 읽고 쓰기)",
    "파이썬 가상 환경 (venv)"
]

python_basics_100 = [
    # 변수 및 데이터 타입
    "변수 선언 및 할당",
    "데이터 타입 (int, float, str, bool)",
    "type() 함수로 데이터 타입 확인",
    "id() 함수로 메모리 주소 확인",
    "isinstance() 함수로 데이터 타입 검사",
    "변수명 규칙과 네이밍 컨벤션",
    
    # 입출력
    "print() 함수 사용법",
    "input() 함수로 사용자 입력 받기",
    "print()에서 sep, end 매개변수 사용",
    
    # 연산자
    "산술 연산자 (+, -, *, /, //, %, **)",
    "할당 연산자 (=, +=, -=, *=, /= 등)",
    "비교 연산자 (==, !=, >, <, >=, <=)",
    "논리 연산자 (and, or, not)",
    "비트 연산자 (&, |, ^, ~, <<, >>)",
    
    # 문자열 (String)
    "문자열 선언 및 사용",
    "문자열 인덱싱 및 슬라이싱",
    "문자열 포매팅 (f-string, format(), % 연산자)",
    "문자열 내장 메서드 (strip, replace, split, join 등)",
    "len() 함수로 문자열 길이 구하기",
    "이스케이프 문자 (\\n, \\t, \\\\ 등)",
    
    # 리스트 (List)
    "리스트 기본 사용법",
    "리스트 인덱싱 및 슬라이싱",
    "리스트 추가 및 삭제 (append, insert, remove, pop)",
    "리스트 정렬 및 역순 (sort, reverse)",
    "리스트 내장 메서드 (index, count 등)",
    "리스트 요소 변경 및 삭제",
    
    # 튜플 (Tuple)
    "튜플 기본 사용법",
    "튜플과 리스트의 차이점",
    "튜플 언패킹",
    
    # 딕셔너리 (Dictionary)
    "딕셔너리 기본 사용법",
    "딕셔너리 키-값 추가 및 삭제",
    "딕셔너리 내장 메서드 (keys, values, items, get)",
    "딕셔너리 컴프리헨션",
    
    # 집합 (Set)
    "집합 기본 사용법",
    "집합 연산 (합집합, 교집합, 차집합)",
    
    # 조건문
    "조건문 (if, elif, else)",
    "삼항 연산자",
    
    # 반복문
    "반복문 (for, while)",
    "range() 함수 사용법",
    "enumerate() 함수 사용법",
    "break와 continue",
    
    # 리스트 컴프리헨션 및 기타
    "리스트 컴프리헨션",
    "zip() 함수 활용",
    "map()과 filter() 함수 활용",
    "리스트 정렬 (sorted, key 매개변수 활용)",
    "이진 탐색 (bisect 모듈)",
    
    # 함수 (Function)
    "함수 정의 및 호출 (def)",
    "함수의 기본 매개변수 및 반환값",
    "가변 인자 (*args, **kwargs)",
    "람다 함수 (lambda)",
    
    # 예외 처리 (Exception Handling)
    "예외 처리 (try, except, finally)",
    "예외 발생시키기 (raise)",
    
    # 파일 입출력
    "파일 입출력 (open, read, write)",
    "with문을 활용한 파일 처리",
    "json 모듈 활용 (json 파일 읽고 쓰기)",
    
    # 클래스 및 객체 지향 프로그래밍
    "클래스 및 객체 기본 개념",
    "생성자 및 소멸자 (__init__, __del__)",
    "클래스 변수와 인스턴스 변수",
    "메서드 정의 및 호출",
    "self 키워드의 의미",
    "상속 및 오버라이딩",
    "다형성과 추상 클래스",
    "특수 메서드 (__str__, __repr__, __len__ 등)",
    
    # 모듈 및 패키지
    "모듈 임포트 (import, from, as)",
    "표준 라이브러리 사용 (math, datetime, random 등)",
    "외부 패키지 설치 및 사용 (pip)",
    
    # 데이터 처리 및 정렬
    "리스트, 딕셔너리, 튜플의 반복문 활용",
    "sorted() 함수 활용",
    "Counter 클래스 활용 (collections 모듈)",
    
    # 기타 고급 문법
    "데코레이터 (@staticmethod, @classmethod)",
    "제너레이터와 yield",
    "이터레이터 패턴",
    
    # 멀티스레딩 및 병렬 처리
    "threading 모듈을 활용한 멀티스레딩",
    "multiprocessing 모듈을 활용한 병렬 처리",
    
    # 네트워크 및 웹 관련
    "requests 모듈을 활용한 HTTP 요청",
    "BeautifulSoup을 활용한 웹 크롤링",
    
    # 데이터 분석 및 시각화
    "numpy 기본 사용법",
    "pandas 기본 사용법",
    "matplotlib을 활용한 데이터 시각화",
    
    # 기타
    "Python 가상 환경 (venv)",
    "Jupyter Notebook 사용법",
    "frozendict (immutable dictionary) 사용법",
    "itertools 모듈 활용",
    "functools 모듈 활용"
]
topic = []
question = []
_comment = []

for instruction in python_basics_100:
    q = gen(instruction)
    t = topic_gen(q)
    c = comment(q)
    topic.append(t)
    question.append(str(q))
    _comment.append(c)
    
df_q = pd.Series(np.array(question))
df_t = pd.Series(np.array(topic))
df_c = pd.Series(np.array(_comment))

# df_t와 df_q를 하나의 데이터프레임으로 결합
df = pd.DataFrame({"topic": df_t, "question": df_q, "comment":df_c})

# CSV 파일로 저장 (옵션)
df.to_csv("data_custom.csv", index=False, encoding='utf-8')

print("데이터프레임 병합 완료!")

