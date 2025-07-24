from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    GROQ_API_KEY="your_actual_groq_api_key_here"
)

resp = llm.invoke("Hello, what causes PCOS?")
print(resp.content)