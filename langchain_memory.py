import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI  # The main class for interacting with Google Generative AI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.globals import set_debug
from langchain.chains.conversation.base import ConversationChain
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain.memory import ConversationBufferMemory
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


mensagens = [
        "Quero visitar um lugar no Brasil famoso por suas praias e cultura. Pode me recomendar?",
        "Qual é o melhor período do ano para visitar em termos de clima?",
        "Quais tipos de atividades ao ar livre estão disponíveis?",
        "Alguma sugestão de acomodação eco-friendly por lá?",
        "Cite outras 20 cidades com características semelhantes às que descrevemos até agora. Rankeie por mais interessante, incluindo no meio ai a que você já sugeriu.",
        "Na primeira cidade que você sugeriu lá atrás, quero saber 5 restaurantes para visitar. Responda somente o nome da cidade e o nome dos restaurantes.",
]

longa_conversa = ""
for mensagem in mensagens:
    longa_conversa += f"Usuário: {mensagem}\n"
    longa_conversa += f"IA: "
    modelo = PromptTemplate(template=longa_conversa, input_variables=[""])
    cadeia = modelo | LLM | StrOutputParser()
    resposta = cadeia.invoke(input={})
    longa_conversa += resposta + "\n"
    print(longa_conversa)
    
memory = ConversationBufferMemory()


conversation = ConversationChain(llm=LLM, 
                                 verbose=True,
                                 memory=memory)