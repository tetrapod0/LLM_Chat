from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from .model import llm

prompt = ChatPromptTemplate.from_template(
    "Don't say anything else and Translate following sentences into Korean:\n{eng_input}")

chain = prompt | llm | StrOutputParser()



