from preprocess_text import clean_text

from gensim.models import Word2Vec, KeyedVectors
import os
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

import openai

openai.api_key = os.environ.get("OPENAI_API_KEY")
openai.organization = os.environ.get("OPENAI_API_ORG")


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
EMBEDDING_FILE = f"{ROOT_DIR}/embeddings/GoogleNews-vectors-negative300.bin.gz"


def w2v_average_embedding(col):
    cleaned = [clean_text(line) for line in col]

    # corpus = []
    # for words in cleaned:
    #     corpus.append(words.split())

    google_word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
    # google_model = Word2Vec(vector_size = 300, window=5, min_count = 2, workers = -1)
    # google_model.build_vocab(corpus)
    # google_model.word(EMBEDDING_FILE, lockf=1.0, binary=True)
    # google_model.train(corpus, total_examples=google_model.corpus_count, epochs = 5)

    word_embeddings = []

    # Reading the each book description
    for line in cleaned:
        avgword2vec = None
        count = 0
        for word in line.split():
            if word in google_word2vec:
                count += 1
                if avgword2vec is None:
                    avgword2vec = google_word2vec[word]
                else:
                    avgword2vec = avgword2vec + google_word2vec[word]

        if avgword2vec is not None:
            avgword2vec = avgword2vec / count

            word_embeddings.append(avgword2vec)

    return np.vstack(word_embeddings)


def query_sentence_embedding(inputs, model="text-embedding-ada-002", batch_size=16):
    inputs = [input.replace("\n", " ") for input in inputs]
    inputs = [input if input != "" else "no infomation" for input in inputs]
    embeddings = []
    for i in tqdm(range(0, len(inputs), batch_size), desc="querying OpenAI API"):
        try:
            embeddings_batch = openai.Embedding.create(
                input=inputs[i : i + batch_size], model=model
            )["data"]
        except Exception as e:
            print(e)
        embeddings_batch = [embedding["embedding"] for embedding in embeddings_batch]
        embeddings.extend(embeddings_batch)
    return np.vstack(embeddings)


def sentence_embedding(col, model="text-embedding-ada-002"):
    return query_sentence_embedding(col, model=model, batch_size=16)


if __name__ == "__main__":
    e = sentence_embedding(["I love you", "I love you"])
    print(e)
    print(len(e))
