import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI  # Classe principal para interagir com Google Generative AI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.globals import set_debug
from langchain.chains.conversation.base import ConversationChain
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain.memory import ConversationSummaryMemory

# Carrega variáveis de ambiente do arquivo .env (como a chave da API)
load_dotenv()
# Ativa o modo debug para mostrar logs detalhados do LangChain
set_debug(True)

# Define um modelo Pydantic para representar um destino de viagem
class Destination(BaseModel):
    """
    Representa um destino de viagem com nome da cidade e motivo para visitá-la.
    """
    city: str = Field("Cidade a visitar")
    reason: str = Field("Motivo pelo qual é interessante visitar essa cidade")

# Parâmetros de exemplo para a conversa (não utilizados diretamente no código)
numero_de_dias = 7
numero_de_criancas = 2
atividade = "praia"

# Inicializa o modelo de linguagem do Google Gemini usando a chave da API
LLM = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("API_KEY"))
# Parser para saída em string
parser = StrOutputParser()
# Parser para saída em JSON usando o modelo Destination
json_parser = JsonOutputParser(pydantic_object=Destination)

# Lista de mensagens simulando uma conversa sobre destinos de viagem
mensagens = [
        "Quero visitar um lugar no Brasil famoso por suas praias e cultura. Pode me recomendar?",
        "Qual é o melhor período do ano para visitar em termos de clima?",
        "Quais tipos de atividades ao ar livre estão disponíveis?",
        "Alguma sugestão de acomodação eco-friendly por lá?",
        "Cite outras 20 cidades com características semelhantes às que descrevemos até agora. Rankeie por mais interessante, incluindo no meio ai a que você já sugeriu.",
        "Na primeira cidade que você sugeriu lá atrás, quero saber 5 restaurantes para visitar. Responda somente o nome da cidade e o nome dos restaurantes.",
]

# Cria uma memória de resumo de conversa para armazenar o contexto
memory = ConversationSummaryMemory(llm=LLM)
# Inicializa a cadeia de conversação com o modelo, memória e modo verboso
conversation = ConversationChain(llm=LLM, 
                                 verbose=True,
                                 memory=memory)
# Executa a conversa, enviando cada mensagem e imprimindo a resposta
for mensagem in mensagens:
    resposta = conversation.predict(input=mensagem)
    print(resposta)
# Exibe o conteúdo atual da memória da conversa (resumo do histórico)
print(memory.load_memory_variables({}))