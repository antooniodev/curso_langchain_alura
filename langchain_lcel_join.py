import os
from dotenv import load_dotenv
from operator import itemgetter
from langchain_google_genai import ChatGoogleGenerativeAI  # Classe principal para interagir com o Google Generative AI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.globals import set_debug
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()  # Carrega variáveis de ambiente do arquivo .env
set_debug(True)  # Ativa o modo debug para ver detalhes do processamento

# Define um modelo de destino de viagem com cidade e motivo
class Destination(BaseModel):
    city: str = Field("Cidade a visitar")
    reason: str = Field("Motivo pelo qual é interessante visitar essa cidade")

# Parâmetros de exemplo
numero_de_dias = 7
numero_de_criancas = 2
atividade = "praia"

# Inicializa o modelo de linguagem do Google Gemini
LLM = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("API_KEY"))

# Parser para saída em JSON usando o modelo Destination
json_parser = JsonOutputParser(pydantic_object=Destination)

# Prompt para sugerir uma cidade baseada no interesse
model_city = PromptTemplate(
    template="""Sugira uma cidade dado meu interesse  por {interesse}.
    {output_formatting}
    """,
    input_variables=["interesse"],  # Variável a ser substituída pelo interesse do usuário
    partial_variables={"output_formatting": json_parser.get_format_instructions()},  # Instruções de formatação do parser
)

# Prompt para sugerir restaurantes populares na cidade
model_restaurants = ChatPromptTemplate.from_template(
    "Sugira restaurantes populares entre locais em {city}"
)

# Prompt para sugerir atividades culturais na cidade
model_culture = ChatPromptTemplate.from_template(
    "Sugira atividades e locais culturais em {city}"
)

# Prompt final para juntar todas as informações em um texto coeso
model_final = ChatPromptTemplate.from_messages(
    [
        ("ai", "Sugestão de viagem para a cidade: {city}."),
        ("ai", "Restaurantes que você não pode perder: {restaurants}."),
        ("ai", "Atividades e locais culturais recomendados: {culture_local}."),
        ("human", "Combine as informações das cadeias anteriores em 2 parágrafos coerentes.")
    ]
)

# Define cada parte da cadeia de prompts
parte_1 = model_city | LLM | json_parser
parte_2 = model_restaurants | LLM | StrOutputParser()
parte_3 = model_culture | LLM | StrOutputParser()
parte_4 = model_final | LLM | StrOutputParser()

# Junta tudo em uma cadeia: cidade -> restaurantes/cultura -> texto final
cadeia = (
    parte_1 |
    {
        "restaurants": parte_2,
        "culture_local": parte_3,
        "city": itemgetter("city")
    }
    | parte_4
)

# Executa a cadeia com o interesse "praia"
result = cadeia.invoke({"interesse": "praia"})

print(result)  # Mostra o resultado final
