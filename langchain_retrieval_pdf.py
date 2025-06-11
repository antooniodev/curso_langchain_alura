import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI  # Classe principal para interagir com Google Generative AI
from langchain_core.output_parsers import StrOutputParser
from langchain.globals import set_debug
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# Ativa o modo debug para saída detalhada
set_debug(True)

# Inicializa o modelo de linguagem do Google Gemini com a chave da API
LLM = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("API_KEY"))

# Inicializa o parser para saída em string
parser = StrOutputParser()

# Carrega o documento em pdf para ser usado como base de conhecimento
carregadores = [
    PyPDFLoader("GTB_standard_Nov23.pdf"),
    PyPDFLoader("GTB_gold_Nov23.pdf"),
    PyPDFLoader("GTB_platinum_Nov23.pdf"),
    ]
documentos = []
for carregador in carregadores:
    documentos.extend(carregador.load())

# Inicializa o divisor de texto para criar segmentos de até 1000 caracteres, com sobreposição de 200 caracteres entre eles
quebrador = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
textos = quebrador.split_documents(documentos)

# Cria embeddings (vetores numéricos) para os textos usando o modelo do Google
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("API_KEY"))

# Cria um banco vetorial FAISS a partir dos textos e embeddings
db = FAISS.from_documents(textos, embeddings)

# Cria uma cadeia de perguntas e respostas baseada em recuperação de contexto
qa_chain = RetrievalQA.from_chain_type(LLM, retriever=db.as_retriever())

# Define a pergunta a ser feita ao sistema
pergunta = "Como devo proceder caso tenha um item comprado roubado"

# Executa a cadeia de QA com a pergunta e obtém o resultado
resultado = qa_chain.invoke({"query": pergunta})

# Exibe o resultado da resposta
print(resultado)