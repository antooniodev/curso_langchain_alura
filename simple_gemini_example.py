from google import genai
numero_de_dias = 7
numero_de_criancas = 2
atividade = "praia"

prompt = f"Crie um roteiro de viagem de {numero_de_dias} dias, para uma família com {numero_de_criancas} crianças que gostam de {atividade}."
print(prompt)

cliente = genai.Client(api_key="")

resposta = cliente.models.generate_content(
    model="gemini-2.0-flash",
    contents=["You are a helpful assistant.", prompt]
)

# print(resposta)

roteiro_viagem = resposta.candidates[0].content.parts[0].text
print(roteiro_viagem)