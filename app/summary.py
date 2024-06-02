from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from .model import llm

prompt = ChatPromptTemplate.from_messages([
    ('system', "You must summarize the User's sentences tremendously. You always answer into Korean. Don't say anything else."),
    ('user', "'''{input}'''"),    
])

chain = prompt | llm | StrOutputParser()



