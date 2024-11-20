import os
import numpy as np
import faiss
import pandas as pd
import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

# Certifique-se de ter baixado os pacotes necessários
nltk.download('punkt')



import os
import subprocess
import sys

def install_requirements():
    requirements_file = "requirements.txt"
    if os.path.exists(requirements_file):
        try:
            # Executa a instalação das dependências no requirements.txt
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
            print("Dependências instaladas com sucesso!")
        except subprocess.CalledProcessError as e:
            print(f"Erro ao instalar dependências: {e}")
    else:
        print(f"Arquivo {requirements_file} não encontrado!")

# Configuração da chave de API
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Função para pré-processar o texto
def preprocess_text(text):
    sentences = sent_tokenize(text)
    return sentences

# Função para dividir o texto em chunks
def split_text(text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_text(text)
    return texts

# Função para criar embeddings e indexação com FAISS
def create_embeddings(texts):
    embeddings = OpenAIEmbeddings()
    doc_embeddings = embeddings.embed_documents(texts)
    dimension = len(doc_embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(doc_embeddings))
    return embeddings, index

# Função para buscar os documentos relevantes
def search_docs(query, embeddings, index, texts, k=5):
    query_embedding = embeddings.embed_query(query)
    distances, indices = index.search(np.array([query_embedding]), k)
    results = [texts[i] for i in indices[0]]
    return results

# Função para gerar a resposta usando o modelo atualizado
def generate_answer(messages, embeddings, index, texts):
    # Obter a última mensagem do usuário
    question = messages[-1]["content"]
    context = search_docs(question, embeddings, index, texts)
    context_str = "\n\n".join(context)
    
    # Preparar as mensagens para a API OpenAI
    api_messages = [
        {"role": "system", "content": "Você é um assistente que responde perguntas sobre o Vestibular da Unicamp 2025 com base no edital oficial. Use as informações fornecidas para responder à pergunta do usuário."},
    ]
    
    # Adicionar mensagens anteriores, limitando o tamanho total
    previous_messages = messages[-3:] if len(messages) >= 3 else messages
    for msg in previous_messages[:-1]:
        api_messages.append(msg)
    
    # Adicionar a última mensagem do usuário com o contexto
    last_user_message = messages[-1]
    last_user_message_with_context = {
        "role": last_user_message["role"],
        "content": f"Contexto:\n{context_str}\n\nPergunta:\n{last_user_message['content']}"
    }
    api_messages.append(last_user_message_with_context)
    
    # Chamar a API OpenAI
    chat_completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=api_messages,
        max_tokens=200,
        temperature=0.7
    )
    answer = chat_completion.choices[0].message.content.strip()
    return answer

# Função principal do aplicativo Streamlit
def main():
    # Chamar a função no início do script
    install_requirements()
    
    st.title("Chatbot Vestibular Unicamp 2025")
    
    st.write("Bem-vindo ao chatbot que responde suas dúvidas sobre o Vestibular da Unicamp 2025!")
    
    if 'texts' not in st.session_state:
        with st.spinner('Processando o edital, por favor aguarde...'):
            with open('Normas.txt', 'r', encoding='utf-8') as file:
                edital_text = file.read()
            sentences = preprocess_text(edital_text)
            texts = split_text(edital_text)
            embeddings, index = create_embeddings(texts)
            st.session_state.texts = texts
            st.session_state.embeddings = embeddings
            st.session_state.index = index
    else:
        texts = st.session_state.texts
        embeddings = st.session_state.embeddings
        index = st.session_state.index
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Exibir mensagens anteriores
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Entrada de chat
    question = st.chat_input("Digite sua pergunta sobre o vestibular:")
    
    if question:
        # Adicionar a mensagem do usuário ao estado da sessão
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        
        # Gerar a resposta do assistente
        with st.spinner('Gerando resposta...'):
            try:
                answer = generate_answer(st.session_state.messages, embeddings, index, texts)
                # Adicionar a resposta do assistente ao estado da sessão
                st.session_state.messages.append({"role": "assistant", "content": answer})
                with st.chat_message("assistant"):
                    st.markdown(answer)
            except Exception as e:
                st.error(f"Ocorreu um erro: {e}")

if __name__ == '__main__':
    main()
