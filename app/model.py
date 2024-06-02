from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
    temperature=0.1,
)