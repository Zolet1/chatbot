import re
import PyPDF2
import sys  # Para receber o argumento da linha de comando

def extract_text_from_pdf_cleaned(pdf_path):
    # Regex para identificar os trechos a serem removidos
    regex = r"\d{2}/\d{2}/\d{4}, \d{2}:\d{2} Procuradoria Geral - Normas\nhttps://www\.pg\.unicamp\.br/norma/\d+/0 \d{1,2}/71"
    
    text = ''
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    
    # Remover os trechos encontrados pelo regex
    cleaned_text = re.sub(regex, '', text)
    
    # Garantir que o arquivo seja criado, se não existir
    try:
        with open("Normas.txt", 'x', encoding='utf-8') as txt_file:  # 'x' para criar o arquivo
            txt_file.write("")  # Inicializa com conteúdo vazio
    except FileExistsError:
        pass  # Arquivo já existe, nada a fazer

    # Salvar o texto extraído no arquivo
    with open("Normas.txt", 'w', encoding='utf-8') as txt_file:
        txt_file.write(cleaned_text)
    
    return cleaned_text

# Receber o nome do arquivo PDF como parâmetro
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python script.py <caminho_do_pdf>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    texto_limpo = extract_text_from_pdf_cleaned(pdf_path)
    print("Texto extraído e salvo em 'Normas.txt'")
