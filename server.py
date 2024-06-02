from fastapi import FastAPI
from langserve import add_routes

from typing import List, Union
from langserve.pydantic_v1 import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.model import llm
from app.vanilla import chain as vanilla_chain
from app.translate import chain as translate_chain
from app.summary import chain as summary_chain
from app.chat import chain as chat_chain

app = FastAPI()
# app = FastAPI(
#   title="LangChain Server",
#   version="1.0",
#   description="A simple API server using LangChain",
# )

add_routes(app, llm, path="/llm",)
add_routes(app, vanilla_chain, path="/vanilla",)
add_routes(app, translate_chain, path="/translate",)
add_routes(app, summary_chain, path="/summary",)


class InputChat(BaseModel): # 변수이름 템플릿이랑 같게.
    messsages1: List[Union[HumanMessage, AIMessage, SystemMessage]] = Field(
        ...,
        description="The chat messages representing the current conversation.",
    )

add_routes(
    app,
    chat_chain.with_types(input_type=InputChat),
    path="/chat",
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True,
    playground_type="chat",
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)