import re
import unicodedata

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

FOLDER = "PRE_16_tokenizacion"
FILES = ["file1.txt", "file2.txt", "file3.txt"]

RESOURCES = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]


def download_resources():
    for resource in RESOURCES:
        nltk.download(resource, quiet=True)


def normalize(text):
    text = text.lower()
    # Los textos traen simbolos tipograficos (guiones largos, comillas
    # curvas, letras griegas) que se descartan al pasar por ASCII
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    # Los guiones y barras separan palabras compuestas ("non-precious")
    text = re.sub(r"[-/]", " ", text)
    return text


def tokenize(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    tokens = word_tokenize(normalize(text))
    # Se conservan unicamente las palabras: se eliminan signos de puntuacion,
    # numeros, palabras de una sola letra y stopwords
    tokens = [token for token in tokens if token.isalpha() and len(token) > 1]
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens


def pregunta_01():
    download_resources()

    for filename in FILES:
        with open(f"{FOLDER}/data/{filename}", "r", encoding="utf-8") as file:
            text = file.read()

        tokens = tokenize(text)

        print(f"{filename}: {len(tokens)} tokens, {len(set(tokens))} unicos")

        with open(
            f"{FOLDER}/submission/{filename}", "w", encoding="utf-8"
        ) as file:
            file.write("\n".join(tokens))
            file.write("\n")


if __name__ == "__main__":
    pregunta_01()
