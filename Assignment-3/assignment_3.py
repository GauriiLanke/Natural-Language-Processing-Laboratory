# Assignment 3: Bag-of-Words & TF-IDF Vectorization

import os
import json
import numpy as np
from typing import List, Tuple
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Create Bag-of-Words matrix from scratch
def bow_from_scratch(corpus: List[str]) -> Tuple[List[str], np.ndarray]:

    # Create sorted vocabulary of unique lowercase words
    vocab = sorted(list(set(
        w.lower() for doc in corpus for w in doc.split()
    )))

    word_to_idx = {w: i for i, w in enumerate(vocab)}

    # Initialize BoW matrix
    matrix = np.zeros((len(corpus), len(vocab)), dtype=int)

    # Count word frequency in each document
    for row_idx, doc in enumerate(corpus):
        for word in doc.split():
            w_clean = word.lower()
            if w_clean in word_to_idx:
                matrix[row_idx, word_to_idx[w_clean]] += 1

    return vocab, matrix


# Calculate TF-IDF from scratch
def tfidf_from_scratch(corpus: List[str]) -> Tuple[List[str], np.ndarray]:

    vocab, bow = bow_from_scratch(corpus)
    N = len(corpus)

    # Calculate Term Frequency (TF)
    doc_lengths = bow.sum(axis=1, keepdims=True)
    tf = bow / np.maximum(doc_lengths, 1)

    # Calculate Document Frequency and IDF
    df = (bow > 0).sum(axis=0)
    idf = np.log((N + 1) / (df + 1)) + 1

    # Calculate TF-IDF
    tfidf = tf * idf

    # L2 normalization
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    tfidf_normalized = np.where(
        norms > 0,
        tfidf / norms,
        tfidf
    )

    return vocab, tfidf_normalized


# Detect plagiarism using Cosine Similarity
def run_plagiarism_check(
    corpus: List[str],
    doc_names: List[str],
    threshold: float = 0.70
):

    # Convert documents into TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus).toarray()

    # Calculate pairwise cosine similarity
    sim_matrix = cosine_similarity(tfidf_matrix)

    print(
        f"{'Document A':<20} | "
        f"{'Document B':<20} | "
        f"{'Cosine Sim':<12} | "
        f"{'Status':<15}"
    )
    print("-" * 75)

    # Compare every pair of documents
    for i in range(len(doc_names)):
        for j in range(i + 1, len(doc_names)):

            score = sim_matrix[i, j]

            # Flag documents whose similarity is >= 70%
            status = (
                "FLAGGED PLAGIARISM"
                if score >= threshold
                else "CLEAN"
            )

            print(
                f"{doc_names[i]:<20} | "
                f"{doc_names[j]:<20} | "
                f"{score:.4f}       | "
                f"{status:<15}"
            )


def main():

    print("=" * 60)
    print("ASSIGNMENT 3: BAG-OF-WORDS & TF-IDF VECTORIZATION")
    print("=" * 60)

    # Locate the JSON dataset in the same folder as this file
    data_path = os.path.join(
        os.path.dirname(__file__),
        "academic_submissions.json"
    )

    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        return

    # Load JSON dataset
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    corpus = [item["content"] for item in data]
    doc_ids = [item["id"] for item in data]

    # 1. TF-IDF implementation from scratch
    vocab_scratch, tfidf_scratch = tfidf_from_scratch(corpus)
    print(f"\nVocabulary Size (From Scratch): {len(vocab_scratch)} words")
    print(f"TF-IDF Matrix Shape: {tfidf_scratch.shape}")

    # 2. Verify using Scikit-learn
    sklearn_vec = TfidfVectorizer()
    tfidf_sklearn = sklearn_vec.fit_transform(corpus).toarray()
    print(f"Scikit-Learn TF-IDF Matrix Shape: {tfidf_sklearn.shape}")

    # 3. Plagiarism detection using Cosine Similarity
    print("\n--- Academic Integrity Plagiarism Audit Report ---")
    run_plagiarism_check(corpus, doc_ids, threshold=0.70)


if __name__ == "__main__":
    main()