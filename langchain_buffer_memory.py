import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI  # Classe principal para interagir com Google Generative AI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.globals import set_debug
from langchain.chains.conversation.base import ConversationChain
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain.memory import ConversationBufferMemory

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()
# Ativa o modo debug para saída detalhada
set_debug(True)

# Define um modelo de dados para o destino usando Pydantic
class Destination(BaseModel):
    city: str = Field("Cidade a visitar")
    reason: str = Field("Motivo pelo qual é interessante visitar essa cidade")

# Parâmetros de exemplo para a conversa
numero_de_dias = 7
numero_de_criancas = 2
atividade = "praia"

# Inicializa o modelo de linguagem do Google Gemini com a chave da API
LLM = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("API_KEY"))
# Parser para saída em string
parser = StrOutputParser()
# Parser para saída em JSON baseada no modelo Destination
json_parser = JsonOutputParser(pydantic_object=Destination)

# Lista de mensagens simulando uma conversa
mensagens = [
        "Quero visitar um lugar no Brasil famoso por suas praias e cultura. Pode me recomendar?",
        "Qual é o melhor período do ano para visitar em termos de clima?",
        "Quais tipos de atividades ao ar livre estão disponíveis?",
        "Alguma sugestão de acomodação eco-friendly por lá?",
        "Cite outras 20 cidades com características semelhantes às que descrevemos até agora. Rankeie por mais interessante, incluindo no meio ai a que você já sugeriu.",
        "Na primeira cidade que você sugeriu lá atrás, quero saber 5 restaurantes para visitar. Responda somente o nome da cidade e o nome dos restaurantes.",
]

# Inicializa a memória de buffer para manter o histórico da conversa
memory = ConversationBufferMemory()
# Cria a cadeia de conversação com o modelo, memória e modo verboso
conversation = ConversationChain(llm=LLM, 
                                 verbose=True,
                                 memory=memory)
# Executa a conversa, enviando cada mensagem e imprimindo a resposta
for mensagem in mensagens:
    resposta = conversation.predict(input=mensagem)
    print(resposta)