import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI  # The main class for interacting with Google Generative AI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tracers.stdout import ConsoleCallbackHandler
from langchain.globals import set_debug
load_dotenv()
set_debug(True)  # Enable debug mode for detailed output


numero_de_dias = 7
numero_de_criancas = 2
atividade = "praia"

LLM = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("API_KEY"))
parser = StrOutputParser()

#The code substitutes the LLMChain method to create a SimpleSequentialChain:
# Create a ChatPromptTemplate for the Google Generative AI model
model_city = ChatPromptTemplate.from_template(
        "Sugira uma cidade dado meu interesse  por {interesse}. A sua saída deve ser somente o nome da cidade. Cidade: ",
)
chain_1 = model_city | LLM | parser

model_restaurants = ChatPromptTemplate.from_template(
    "Sugira restaurantes populares entre locais em {cidade}"
)
chain_2 = model_restaurants | LLM | parser

model_culture = ChatPromptTemplate.from_template(
    "Sugira atividades e locais culturais em {cidade}"
)

chain_3 = model_culture | LLM | parser

# Create a SquentialChain to combine the prompts and the model
step1 = chain_1.invoke("praia")
step2 = chain_2.invoke(step1)
step3 = chain_3.invoke(step2)

print(step3)

