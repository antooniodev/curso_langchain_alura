# This code is a simple example of using LangChain with Google Generative AI to create a travel itinerary.
# Ensure you have the required packages installed:
from langchain_google_genai import ChatGoogleGenerativeAI # The main class for interacting with Google Generative AI
from langchain.prompts import PromptTemplate # For creating prompts
from langchain_core.prompts import ChatPromptTemplate # For creating chat prompts
import os
from dotenv import load_dotenv
load_dotenv()
numero_de_dias = 7
numero_de_criancas = 2
atividade = "praia"

#Create a prompt template for generating a travel itinerary
modelo_do_prompt = PromptTemplate.from_template(
    "Crie um roteiro de viagem de {dias} dias, para uma família com {criancas} crianças que gostam de {atividade}."
)

# Create the prompt using the template and the variables
prompt = modelo_do_prompt.format(dias=numero_de_dias, criancas=numero_de_criancas, atividade=atividade)
print(prompt)

# Create a ChatPromptTemplate for the Google Generative AI model
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um assistente de viagem que cria roteiros personalizados."),
        ("user", "{prompt}"),
        ("system", "Responda com um roteiro detalhado, incluindo atividades, locais e horários."),
        ("user", "Eu gostaria que o roteiro visitasse praias do Brasil.")
    ]
)

chat_template = chat_prompt.format(prompt=prompt) # Format the prompt for the chat model
## Initialize the Google Generative AI client with your API key
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("API_KEY"))

resposta = llm.invoke(chat_template) # Invoke the model with the prompt
print(resposta.content)

    