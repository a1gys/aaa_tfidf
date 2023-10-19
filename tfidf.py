from typing import List, Iterable
import math


class CountVectorizer:

    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase
        self._vocab = {}

    def fit_transform(self, texts: Iterable) -> List[List[int]]:
        words = []

        for text in texts:
            if self.lowercase:
                text = text.lower()
            for word in text.split():
                if word not in words:
                    words.append(word)

        self._vocab = {word: i for i, word in enumerate(words)}

        vector = [[0] * len(words) for _ in range(len(texts))]

        for i, text in enumerate(texts):
            if self.lowercase:
                text = text.lower()
            for word in text.split():
                vector[i][self._vocab[word]] += 1

        return vector

    def get_feature_names(self) -> List[str]:
        return list(self._vocab.keys())


def tf_transform(matrix: List[List[int]]) -> List[List[int]]:
    """
    Term frequency function
    finds the proportion of the occurences of the word in the document
    """

    tf_matrix = [[0] * len(matrix[0]) for _ in range(len(matrix))]

    for i, vector in enumerate(matrix):
        total = sum(vector)
        for j, num in enumerate(vector):
            tf_matrix[i][j] = round(num / total, 3)

    return tf_matrix


def idf_transform(matrix: List[List[int]]) -> List[int]:
    """
    Inverse document frequency
    """
    idf_vector = [0] * len(matrix[0])
    total_doc_num = len(matrix)

    for j in range(len(matrix[0])):
        word_doc_num = 0
        for i in range(total_doc_num):
            if matrix[i][j]:
                word_doc_num += 1
        idf_word = math.log((total_doc_num+1)/(word_doc_num+1))+1
        idf_vector[j] = round(idf_word, 1)
    return idf_vector


class TfidfTransformer:
    def __init__(self):
        pass

    def fit_transform(self, count_matrix: List[List[int]]) -> List[List[int]]:
        tf = self.tf_transform(matrix=count_matrix)
        idf = self.idf_transform(matrix=count_matrix)

        tfidf = [[0] * len(count_matrix[0]) for _ in range(len(count_matrix))]

        for i, vector in enumerate(tf):
            for j in range(len(vector)):
                tfidf[i][j] = round(vector[j] * idf[j], 3)

        return tfidf

    def tf_transform(self, matrix: List[List[int]]) -> List[List[int]]:
        """
        Term frequency function
        finds the proportion of the occurences of the word in the document
        """

        tf_matrix = [[0] * len(matrix[0]) for _ in range(len(matrix))]

        for i, vector in enumerate(matrix):
            total = sum(vector)
            for j, num in enumerate(vector):
                tf_matrix[i][j] = round(num / total, 3)

        return tf_matrix

    def idf_transform(self, matrix: List[List[int]]) -> List[int]:
        """
        Inverse document frequency
        """
        idf_vector = [0] * len(matrix[0])
        total_doc_num = len(matrix)

        for j in range(len(matrix[0])):
            word_doc_num = 0
            for i in range(total_doc_num):
                if matrix[i][j]:
                    word_doc_num += 1
            idf_word = math.log((total_doc_num+1)/(word_doc_num+1))+1
            idf_vector[j] = round(idf_word, 1)
        return idf_vector

    @classmethod
    def transform(cls, count_matrix: List[List[int]]):
        transformer = cls()
        return transformer.fit_transform(count_matrix)


class TfidfVectorizer(CountVectorizer):
    def __init__(self, lowercase: bool = True):
        super().__init__(lowercase=lowercase)

    def fit_transform(self, texts: Iterable):
        count_matrix = super().fit_transform(texts=texts)
        tfidf = TfidfTransformer.transform(count_matrix)
        return tfidf

    def tf_transform(self, matrix: List[List[int]]) -> List[List[int]]:
        """
        Term frequency function
        finds the proportion of the occurences of the word in the document
        """

        tf_matrix = [[0] * len(matrix[0]) for _ in range(len(matrix))]

        for i, vector in enumerate(matrix):
            total = sum(vector)
            for j, num in enumerate(vector):
                tf_matrix[i][j] = round(num / total, 3)

        return tf_matrix

    def idf_transform(self, matrix: List[List[int]]) -> List[int]:
        """
        Inverse document frequency
        """
        idf_vector = [0] * len(matrix[0])
        total_doc_num = len(matrix)

        for j in range(len(matrix[0])):
            word_doc_num = 0
            for i in range(total_doc_num):
                if matrix[i][j]:
                    word_doc_num += 1
            idf_word = math.log((total_doc_num+1)/(word_doc_num+1))+1
            idf_vector[j] = round(idf_word, 1)
        return idf_vector


if __name__ == "__main__":
    corpus = ["Crock Pot Pasta Never boil pasta again",
              "Pasta Pomodoro Fresh ingredients Parmesan to taste"]

    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(corpus)
    print(f"Count matrix from CountVectorizer: {count_matrix}")
    print(f"Features from CountVectorizer: {vectorizer.get_feature_names()}")

    transformer = TfidfTransformer()
    tfidf_matrix = transformer.fit_transform(count_matrix)
    print(f"Tfidf matrix from TfidfTransformer: {tfidf_matrix}")

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    print(f"Tfidf matrix from TfidfVectorizer: {tfidf_matrix}")
