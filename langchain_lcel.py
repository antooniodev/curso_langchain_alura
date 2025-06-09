import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI  # Classe principal para interagir com o Google Generative AI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.globals import set_debug
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()
# Ativa o modo debug para ver detalhes do processamento
set_debug(True)

# Define um modelo de dados para o destino sugerido
class Destination(BaseModel):
    city: str = Field("Cidade a visitar")
    reason: str = Field("Motivo pelo qual é interessante visitar essa cidade")

# Parâmetros de exemplo para a viagem
numero_de_dias = 7
numero_de_criancas = 2
atividade = "praia"

# Inicializa o modelo de linguagem do Google com a chave da API
LLM = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("API_KEY"))

# Parser para garantir que a resposta do modelo siga o formato do Destination
json_parser = JsonOutputParser(pydantic_object=Destination)

# Prompt para sugerir uma cidade baseada no interesse informado
model_city = PromptTemplate(
    template="""Sugira uma cidade dado meu interesse  por {interesse}.
    {output_formatting}
    """,
    input_variables=["interesse"],  # Variável que será substituída pelo interesse do usuário
    partial_variables={"output_formatting": json_parser.get_format_instructions()},  # Instruções de formatação para o modelo
)

# Prompt para sugerir restaurantes populares na cidade sugerida
model_restaurants = ChatPromptTemplate.from_template(
    "Sugira restaurantes populares entre locais em {city}"
)

# Prompt para sugerir atividades culturais na cidade sugerida
model_culture = ChatPromptTemplate.from_template(
    "Sugira atividades e locais culturais em {city}"
)

# Define a cadeia de execução:
# 1. Sugere uma cidade (parte_1)
# 2. Usa a cidade sugerida para buscar restaurantes (parte_2) e locais culturais (parte_3)
parte_1 = model_city | LLM | json_parser
parte_2 = model_restaurants | LLM | StrOutputParser()
parte_3 = model_culture | LLM | StrOutputParser()

# Junta tudo em uma cadeia sequencial: cidade -> restaurantes e cultura
cadeia = (parte_1 | {
    "restuarantes": parte_2, 
    "locais_culturais": parte_3
})

# Executa a cadeia com o interesse "praia"
result = cadeia.invoke({"interesse": "praia"})

# Mostra o resultado final
print(result)
