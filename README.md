# LLM_Chat

---

### Step. 0

- 서버 생성을 위해 필요한 모듈들을 설치해준다.
- langserve 까지만 설치했었는데 실행해보니 필요한 것들 더 설치하라고 에러 띄운다.

```
langchain
langchain_openai

langserve
fastapi
uvicorn
sse_starlette
pydantic==1.10.13
```

---

### Step. 1
- app/model.py
- 해당 파일 안에 아래 코드를 넣어 다른 py파일에서도 llm을 사용할 수 있게 만든다.

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
    temperature=0.1,
)
```

---

### Step. 2

- app/vanilla.py
- 테스트를 위해 해당 파일에 아래 코드를 넣는다.
- 대화는 이어지지 않고 단순 질문에 답변만 해준다.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from .model import llm

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful, smart, kind, and efficient AI assistant. You always fulfill the user's requests to the best of your ability. You always answer succinctly. You must answer in Korean."),
    ("user", "{user_input}"),
])
chain = prompt | llm | StrOutputParser()
```

---

### Step. 3

- app/server.py
- 서버 실행을 위한 간단한 코드이다.
- 아래 코드를 실행하여 간단히 테스트 해보자.

```python
from fastapi import FastAPI
from langserve import add_routes
from app.vanilla import chain as vanilla_chain

app = FastAPI()
add_routes(app, vanilla_chain, path="/vanilla",)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
```

- 아래 주소로 들어가서 확인해보자.
- http://localhost:8000/vanilla/playground/
- 아래 처럼 간단한 질문을 던질 수 있다.

<img src=https://github.com/tetrapod0/LLM_Chat/assets/48349693/41dc37a4-d400-4dc7-a791-c6fedede6ea2 width=70%>

---

### Step. 4

- app/server.py
- 이번엔 번역, 요약 기능 페이지를 만들어보자.

```python
from fastapi import FastAPI
from langserve import add_routes
from app.vanilla import chain as vanilla_chain
from app.translate import chain as translate_chain
from app.summary import chain as summary_chain

app = FastAPI()
add_routes(app, vanilla_chain, path="/vanilla",)
add_routes(app, translate_chain, path="/translate",)
add_routes(app, summary_chain, path="/summary",)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
```

- app/translate.py

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from .model import llm

prompt = ChatPromptTemplate.from_template(
    "Don't say anything else and Translate following sentences into Korean:\n{eng_input}")

chain = prompt | llm | StrOutputParser()
```

- app/summary.py

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from .model import llm

prompt = ChatPromptTemplate.from_messages([
    ('system', "You must summarize the User's sentences tremendously. You always answer into Korean. Don't say anything else."),
    ('user', "'''{input}'''"),    
])

chain = prompt | llm | StrOutputParser()
```

- 서버를 실행 후 아래 링크에서 테스트 해볼 수 있다.
- http://localhost:8000/translate/playground/
- http://localhost:8000/summary/playground/

---

### Step. 5

- app/chat.py
- 대화가 이어질려면 비교적 복잡해진다.
- MessagesPlaceholder라는 것을 템플릿에 추가해줘야한다.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from .model import llm

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful, smart, kind, and efficient AI assistant. You always fulfill the user's requests to the best of your ability. You always answer succinctly. You must answer in Korean."),
    MessagesPlaceholder(variable_name='messsages1'),
])
chain = prompt | llm | StrOutputParser()
```

- app/server.py
- 다음과 같이 파일을 수정 후 실행 시켜보자.

```python
from fastapi import FastAPI
from langserve import add_routes

from typing import List, Union
from langserve.pydantic_v1 import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.vanilla import chain as vanilla_chain
from app.translate import chain as translate_chain
from app.summary import chain as summary_chain
from app.chat import chain as chat_chain

app = FastAPI()

add_routes(app, vanilla_chain, path="/vanilla",)
add_routes(app, translate_chain, path="/translate",)
add_routes(app, summary_chain, path="/summary",)


class InputChat(BaseModel): # 클래스변수이름 템플릿이랑 같게.
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
```

- 아래 링크로 들어가면 이제 대화가 이어지는 채팅을 할 수 있다.
- http://localhost:8000/chat/playground/

---

### Extra

- app/server.py
- 아래 코드를 추가하면 RemoteRunnable을 사용할 수 있다.

```python
from app.model import llm
add_routes(app, llm, path="/llm",)
```

- 사실상 llm과 llm2은 같은 것이다.

```python
from langserve import RemoteRunnable

llm2 = RemoteRunnable("http://localhost:8000/llm")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful, smart, kind, and efficient AI assistant."),
    ("user", "{input} 한국어로 대답해줘."),
])
chain = prompt | llm2 | StrOutputParser()
msg = chain.invoke({'input' : '안녕하세요!'})
```

---

### Reference

- https://velog.io/@kwon0koang/%EB%A1%9C%EC%BB%AC%EC%97%90%EC%84%9C-Llama3-%EB%8F%8C%EB%A6%AC%EA%B8%B0
- https://python.langchain.com/v0.1/docs/use_cases/chatbots/quickstart/
- https://medium.com/@amanatulla1606/langserve-deploying-language-models-made-easy-9f6280210ba4
