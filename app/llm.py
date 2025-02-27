from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from typing import List, TypedDict

class State(TypedDict, total= False):
    #context_docs : 검색된 벡터데이터
    original_codes: str
    oneline_code: str
    user_id: str

class LLM:
    def __init__(self):
        load_dotenv()
        self.model = init_chat_model("gpt-4o-mini", model_provider="openai")

        # 프롬프트 템플릿 정의
        self.prompt_initial_template = ChatPromptTemplate.from_messages(
            [
                ("system", "당신은 30년 경력의 파이썬 교육자입니다. 사용자가 입력한 코드를 기반으로 {context_docs} 를 참조해서 한줄 한줄마다 코드의 의미를 설명하는 문장을 생성해주세요. 너무 복잡한 개념을 한꺼번에 설명하기보다는 초보자가 쉽게 따라할 수 있도록 단계적으로 설명하세요."),
                ("user", "{original_codes}")
            ]
        )
        self.prompt_hint_for_fail_template = ChatPromptTemplate.from_messages(
            [
                ("system", "당신은 30년 경력의 파이썬 강사입니다. 학생이 한 줄씩 파이썬 코드를 입력하면 {context_docs}를 참조하고 사용자의 입력이 {original_codes} 의 한줄이라는걸 인지하고 코드의 오류 메시지를 만들어서 초보자가 이해할 수 있도록 쉽게 설명하세요. 어떤 부분에서  문제가 발생했는지 정확하게 짚어 주세요. 초보자가 실수를 방지할 수 있도록 힌트를 제공하세요. 코드 예제를 보여주지 마세요.너무 많은 정보를 한꺼번에 주지 말고, 핵심적인 부분만 짚어 주세요. 학생이 실수를 통해 학습할 수 있도록 유도하세요."),
                ("user", "{oneline_code}")
            ]
        )
        self.prompt_final_explanation_template = ChatPromptTemplate.from_messages(
            [
                ("system","{memory_state}를 참고해서 유저가 앞으로 {original_codes}를 작성할때 자주 실수할수 있는 문제나 중요한점을 짚어서 최종 해설을 해줘")
            ]
        ) 
        # LangGraph 워크플로우 설정
        self.workflow = StateGraph(state_schema=State)

        def initial_interpretation(state: State):
            """벡터 데이터 + 유저 입력 → LLM 호출"""
            #context = "\n".join(state["context_docs"])
            prompt = self.prompt_initial_template.invoke({"original_codes": state["original_codes"], "context_docs": state["context_docs"]})
            response = self.model.invoke(prompt)
            return {"initial_response": response}

        def user_input_interpretation(state: State):
             #context = "\n".join(state["context_docs"])
            prompt = self.prompt_hint_for_fail_template.invoke({"context_docs": state["context_docs"], "oneline_code" : state["oneline_code"], "original_codes" : state["original_codes"]})
            response = self.model.invoke(prompt)
            return {"user_input_response": response}
        
        def final_summary(state: State):
            memory_state = self.memory.load_state(state["user_id"])
            prompt = self.prompt_final_explanation_template.invoke({"memory_state": memory_state, "original_codes" : state["original_codes"]})
            response = self.model.invoke(prompt)
            return {'final explanation': response}

        self.workflow.add_node("initial_interpretation", self.initial_interpretation)
        self.workflow.add_node("user_input_interpretation", self.user_input_interpretation)
        self.workflow.add_node("final_summary", self.final_summary)

        self.workflow.add_edge(START, "initial_interpretation") 

        # MemorySaver를 사용하여 사용자별 대화 이력 저장
        self.memory = MemorySaver() 
        self.app = self.workflow.compile(checkpointer=self.memory)

    def interpret_initial_code(self, user_id: str, original_codes: str, context_docs: list[str]) -> str:
        """사용자별 대화 이력 + RAG 응답 반환"""
        config = {"configurable": {"thread_id": user_id}}
        state = {"context_docs": context_docs, "original_codes": original_codes, "user_id": user_id}
        output = self.app.invoke_step("initial_interpretation",state, config )
        return output["initial_response"]

    def interpret_user_input(self, user_id: str, oneline_code: str, context_docs: list[str]) -> str:
        config = {"configurable": {"thread_id": user_id}}
        state = self.memory.load_state(user_id)
        state["oneline_code"] = oneline_code
        output = self.app.invoke_step("user_input_interpretation",state, config)
        return output["user_input_response"]
    
    def summarize_user_inputs(self, user_id: str) -> str:
        config = {"configurable": {"thread_id": user_id}}
        state = self.memory.load_state(user_id)
        output = self.app.invoke_step("final_summary", state, config)
        return output["final explanation"]
    