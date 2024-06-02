from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from .model import llm

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful, smart, kind, and efficient AI assistant. You always fulfill the user's requests to the best of your ability. You always answer succinctly. You must answer in Korean."),
    # ("user", "{user_input} 한국어로 대답해줘."),
    MessagesPlaceholder(variable_name='messsages1'),
])
chain = prompt | llm | StrOutputParser()



