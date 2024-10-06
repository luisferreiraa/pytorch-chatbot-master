# Importação das bibliotecas necessárias
import os
from flask import Flask, render_template, request, jsonify, send_from_directory
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Inicializa a aplicação Flask
app = Flask(__name__)

# Configuração do dispositivo para utilizar GPU se disponível
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Carrega os intents a partir do arquivo JSON
with open('intents.json', 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)

# Carrega os dados do modelo treinado
FILE = "data.pth"
data = torch.load(FILE)

# Extrai informações do modelo
input_size = data["input_size"]  # Tamanho da entrada
hidden_size = data["hidden_size"]  # Tamanho da camada escondida
output_size = data["output_size"]  # Tamanho da saída
all_words = data['all_words']  # Lista de todas as palavras
tags = data['tags']  # Lista de todas as tags
model_state = data["model_state"]  # Estado do modelo treinado

# Inicializa o modelo de rede neural
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)  # Carrega os pesos do modelo
model.eval()  # Coloca o modelo em modo de avaliação


@app.route('/')  # Rota para a página inicial
def home():
    return render_template('home.html')  # Renderiza o template home.html


@app.route('/cv')  # Rota para download do CV
def get_cv():
    # Envia o ficheiro pdf localizado na página estática
    return send_from_directory(os.path.join(app.root_path, 'static'), 'cv_luis.pdf')


@app.route('/ask', methods=['POST'])    # Rota para processar perguntas
def ask():
    user_input = request.json['question']    # Obtém a pergunta do utilizador do corpo da requesição
    sentence = tokenize(user_input)     # Tokeniza a entrada do utilizador
    X = bag_of_words(sentence, all_words)   # Converte a sentence numa representação de bag-of-words
    X = X.reshape(1, X.shape[0])    # Redimensiona a entrada para ser compatível com a rede
    X = torch.from_numpy(X).to(device)      # Converte para tensor e envia para o dispositivo apropiado

    output = model(X)   # Faz a previsão utilizando o modelo
    _, predicted = torch.max(output, dim=1)     # Obtém a classe prevista

    tag = tags[predicted.item()]    # Extrai a tag correspondente à previsão

    probs = torch.softmax(output, dim=1)    # Aplica o softmax para obter as probabilidades
    prob = probs[0][predicted.item()]       # Obtém a probabilidade da previsão

    # Verifica se a probabilidade da previsão é maior que 0.75
    if prob.item() > 0.75:
        for intent in intents['intents']:   # Percorre os intents para encontrar a resposta
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])   # Escolhe uma resposta aleatória
                return jsonify({'answer': response})            # Retorna a resposta em formato JSON
    else:
        return jsonify({'answer': "Peço desculpa, não entendi."})    # Retorna uma resposta padrão se não entendeu


if __name__ == '__main__':
    app.run(debug=True)     # Executa a aplicação em modo debug
