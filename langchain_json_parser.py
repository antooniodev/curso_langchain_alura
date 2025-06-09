import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI  # The main class for interacting with Google Generative AI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.globals import set_debug
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
load_dotenv()
set_debug(True)  # Enable debug mode for detailed output

class Destination(BaseModel):
    city: str = Field("Cidade a visitar")
    reason: str = Field("Motivo pelo qual é interessante visitar essa cidade")

numero_de_dias = 7
numero_de_criancas = 2
atividade = "praia"

LLM = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("API_KEY"))
parser = StrOutputParser()
json_parser = JsonOutputParser(pydantic_object=Destination)

model_city = PromptTemplate(
        template="""Sugira uma cidade dado meu interesse  por {interesse}.
        {output_formatting}
        """,
        input_variables=["interesse"], # The variable to be replaced in the template, received from the user input
        partial_variables={"output_formatting": json_parser.get_format_instructions()}, # The variable to be replaced in the template, received from the json parser
)

chain_1 = model_city | LLM | parser

# Create a SquentialChain to combine the prompts and the model
step1 = chain_1.invoke("praia")

print(step1)

