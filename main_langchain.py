from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
load_dotenv()
numero_de_dias = 7
numero_de_criancas = 2
atividade = "praia"

prompt = f"Crie um roteiro de viagem de {numero_de_dias} dias, para uma família com {numero_de_criancas} crianças que gostam de {atividade}."
print(prompt)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("API_KEY"))

resposta = llm.invoke(prompt)
print(resposta.content)

    