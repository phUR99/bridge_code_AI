from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import List, TypedDict, Union

class State(TypedDict, total= False):
    context_docs : list[str]
    original_codes: str
    oneline_code: str
    user_id: str
    prompt_model: str
    response: str

class LLM:
    original_codes = "Hello world"
    def __init__(self, codes):
        original_codes = codes
        load_dotenv()
        model = ChatOpenAI(model="gpt-4o-mini")

        # 프롬프트 템플릿 정의
        prompt_initial_template = ChatPromptTemplate.from_messages(
            [
                ("system", "당신은 Python 프로그래밍을 가르치는 교육 도우미입니다. 사용자가 최종적으로 완성해야 할 파이썬 코드를 입력하면, "
                        "각 코드 줄마다 하나의 가이드 문장을 작성해야 합니다. "
                        "가이드는 코드가 수행해야 할 작업을 설명하며, 학생이 직접 코드를 작성할 수 있도록 유도해야 합니다. "
                        "너무 복잡한 개념을 한꺼번에 설명하지 말고, 초보자가 이해하기 쉽게 단계적으로 안내하세요. "
                        "반복 횟수, 변수명, 조건 등은 명확하게 서술하며, 불필요한 코드 설명은 배제하세요. "
                        "만약 코드에 빈 줄이 포함되어 있다면 무시하세요."),
                ("user", "{original_codes}")
            ]
        )

        prompt_hint_for_fail_template = ChatPromptTemplate.from_messages(
            [
                ("system", "당신은 30년 경력의 파이썬 강사입니다. 학생이 한 줄씩 파이썬 코드를 입력하면, {context_docs}를 참조하여 "
                        "해당 코드({oneline_code})가 {original_codes}의 한 줄 중 하나라는 점을 고려하세요. "
                        "코드에서 발생할 수 있는 문제를 초보자가 이해할 수 있도록 쉽게 설명하고, 직접적인 정답을 제공하지 말고 간접적인 힌트를 주세요. "
                        "학생이 실수를 통해 학습할 수 있도록 유도하세요. "
                        "힌트는 지나치게 많은 정보를 제공하지 않도록 주의하며, 핵심적인 부분만 짚어주세요. "
                        "구문 오류(SyntaxError), 논리 오류(Logic Error), 오타 등의 일반적인 실수에 대해 올바른 방향을 제시하세요. "
                        "단, 코드 예제는 절대 제공하지 마세요."),
                ("user", "{oneline_code}")
            ]
        )
        prompt_final_explanation_template = ChatPromptTemplate.from_messagesprompt_study_guide_template = ChatPromptTemplate.from_messages(
            [
                ("system", "당신은 30년 경력의 파이썬 강사입니다. "
                        "지금까지 사용자가 입력한 코드와 그에 대한 피드백을 기억하고 있습니다. "
                        "이전 대화 내용을 참조하여, 사용자가 자주 했던 실수 패턴을 분석하고 "
                        "원래 작성해야 할 코드({original_codes})와 비교해 어떤 점을 주의해야 할지 설명하세요. "
                        "사용자가 앞으로 같은 유형의 코드를 작성할 때 실수하지 않도록 가이드를 제공하세요. "
                        "너무 많은 정보를 한꺼번에 제공하지 말고, 핵심적인 실수 패턴을 정리해서 전달하세요. "
                        "정답을 직접 알려주지 말고, 사용자가 스스로 학습할 수 있도록 유도하세요."),
                ("user", "지금까지의 대화 내용을 참고하여, 내가 자주 했던 실수를 분석하고 앞으로 같은 유형의 코드를 작성할 때 주의해야 할 점을 알려줘.")
            ]
        )
        

        # LangGraph 워크플로우 설정
        self.workflow = StateGraph(state_schema=State)

        def initial_interpretation(state: State):
            """벡터 데이터 + 유저 입력 → LLM 호출"""
            context_docs = ",".join(state["context_docs"])
            if state["prompt_model"] == "initial_model":
                prompt = prompt_initial_template.invoke({"original_codes": state["original_codes"], "context_docs": context_docs})
            elif state["prompt_model"] == "user_input_model":
                prompt = prompt_hint_for_fail_template.invoke({"original_codes": state["original_codes"], "context_docs": context_docs, 'oneline_code': state['oneline_code']})
            else:
                prompt = prompt_final_explanation_template.invoke({'original_codes':state['original_codes']})
            #print(prompt)
            response = model.invoke(prompt)
            return {"response" : response.content}
        '''
        def user_input_interpretation(state: State):
             #context = "\n".join(state["context_docs"])
            prompt = self.prompt_hint_for_fail_template.invoke({"context_docs": state["context_docs"], "oneline_code" : state["oneline_code"], "original_codes" : state["original_codes"]})
            response = self.model.invoke(prompt)
            return {"response": response}
        
        #def final_summary(state: State):
            memory_state = self.memory.load_state(state["user_id"])
            prompt = self.prompt_final_explanation_template.invoke({"memory_state": memory_state, "original_codes" : state["original_codes"]})
            response = self.model.invoke(prompt)
            return {'final explanation': response}
        '''
        self.workflow.add_node("initial_interpretation", initial_interpretation)
        

        self.workflow.add_edge(START, "initial_interpretation") 

        # MemorySaver를 사용하여 사용자별 대화 이력 저장
        self.memory = MemorySaver() 
        self.app = self.workflow.compile(checkpointer=self.memory)

    def interpret_initial_code(self, user_id: str,  prompt_model : str, original_codes:Union[str,None]=original_codes, context_docs: Union[list[str],None]=None, oneline_code: Union[str, None] = None) -> str:
        """사용자별 대화 이력 + RAG 응답 반환"""
        config = {"configurable": {"thread_id": user_id}}
        print(original_codes)
        if prompt_model == "initial_model":
            state = {"context_docs": context_docs, "original_codes": original_codes, "user_id": user_id, "prompt_model" : prompt_model}
        elif prompt_model == "user_input_model":
            state = {            
                "user_id" : user_id,
                "oneline_code" : oneline_code,
                "context_docs" : context_docs,
                "prompt_model" : prompt_model,
                "original_codes" : original_codes
            }
        else:
            state = {"user_id" : user_id, 'prompt_model' : prompt_model, 'original_codes' : original_codes}
        #print(context_docs)
        print(state)
        output = self.app.invoke(state, config)
        print(output)
        return output["response"]

    #def interpret_user_input(self, user_id: str, oneline_code: str, context_docs: list[str]) -> str:
        config = {"configurable": {"thread_id": user_id}}
        state = self.memory.load_state(user_id)
        state["oneline_code"] = oneline_code
        output = self.app.invoke_step("user_input_interpretation",state, config)
        return output["user_input_response"]
    
    #def summarize_user_inputs(self, user_id: str) -> str:
        config = {"configurable": {"thread_id": user_id}}
        state = self.memory.load_state(user_id)
        output = self.app.invoke_step("final_summary", state, config)
        return output["final explanation"]
    