import pymorphy3
import re
from collections import Counter
from typing import List, Dict, Union, Optional
import numpy as np


class TextVectorizer:
    def __init__(self, 
                 use_lemmatization: bool = True,
                 min_word_length: int = 2,
                 remove_stopwords: bool = True,
                 custom_stopwords: Optional[List[str]] = None):
        self.use_lemmatization = use_lemmatization
        self.min_word_length = min_word_length
        self.remove_stopwords = remove_stopwords
        
        self.morph_analyzer = pymorphy3.MorphAnalyzer()
        
        self.stopwords = set([
            'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а',
            'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же',
            'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от',
            'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже',
            'ну', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до',
            'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя',
            'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней',
            'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без', 'будто',
            'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда', 'кто',
            'этот', 'того', 'потому', 'этого', 'какой', 'совсем', 'ним', 'здесь',
            'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас', 'были',
            'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец', 'два',
            'об', 'другой', 'хоть', 'после', 'над', 'больше', 'тот', 'через',
            'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много', 'разве',
            'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 'перед',
            'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более',
            'всегда', 'конечно', 'всю', 'между'
        ])
        
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)
        
        self.vocabulary = {}
        self.vocabulary_size = 0
        
    def preprocess_text(self, text: str) -> List[str]:
        text = text.lower()
        
        text = re.sub(r'[^а-яёa-z\s]', ' ', text)
        
        words = text.split()
        
        processed_words = []
        for word in words:
            if len(word) < self.min_word_length:
                continue
                
            if self.remove_stopwords and word in self.stopwords:
                continue
            
            if self.use_lemmatization:
                parsed_word = self.morph_analyzer.parse(word)[0]
                normalized_word = parsed_word.normal_form
            else:
                normalized_word = word
            
            processed_words.append(normalized_word)
        
        return processed_words
    
    def build_vocabulary(self, texts: List[str]) -> None:
        all_words = []
        
        for text in texts:
            words = self.preprocess_text(text)
            all_words.extend(words)
        
        word_counts = Counter(all_words)
        
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(
            sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        )}
        self.vocabulary_size = len(self.vocabulary)
    
    def vectorize_text(self, text: str) -> np.ndarray:

        if not self.vocabulary:
            raise ValueError("Словарь не построен. Сначала вызовите build_vocabulary.")
        
        words = self.preprocess_text(text)
        
        vector = np.zeros(self.vocabulary_size, dtype=np.float32)
        
        for word in words:
            if word in self.vocabulary:
                idx = self.vocabulary[word]
                vector[idx] += 1
        
        if np.sum(vector) > 0:
            vector = vector / np.sum(vector)
        
        return vector
    
    def vectorize_texts(self, data: List[Dict[str, str]]) -> Dict[str, np.ndarray]:

        texts = [item['text'] for item in data]
        
        self.build_vocabulary(texts)
        
        vectors = {}
        for item in data:
            filename = item['file']
            text = item['text']
            vector = self.vectorize_text(text)
            vectors[filename] = vector
        
        return vectors
    
    def vectorize_with_tfidf(self, data: List[Dict[str, str]]) -> Dict[str, np.ndarray]:

        texts = [item['text'] for item in data]
        
        self.build_vocabulary(texts)
        
        doc_freq = np.zeros(self.vocabulary_size, dtype=np.float32)
        
        for item in data:
            words = set(self.preprocess_text(item['text']))
            for word in words:
                if word in self.vocabulary:
                    idx = self.vocabulary[word]
                    doc_freq[idx] += 1
        
        num_docs = len(data)
        idf = np.log((num_docs + 1) / (doc_freq + 1)) + 1
        
        vectors = {}
        for item in data:
            filename = item['file']
            text = item['text']
            
            tf_vector = np.zeros(self.vocabulary_size, dtype=np.float32)
            words = self.preprocess_text(text)
            
            for word in words:
                if word in self.vocabulary:
                    idx = self.vocabulary[word]
                    tf_vector[idx] += 1
            
            if np.sum(tf_vector) > 0:
                tf_vector = tf_vector / np.sum(tf_vector)
            
            tfidf_vector = tf_vector * idf
            
            norm = np.linalg.norm(tfidf_vector)
            if norm > 0:
                tfidf_vector = tfidf_vector / norm
            
            vectors[filename] = tfidf_vector
        
        return vectors
    
    def get_most_frequent_words(self, top_n: int = 20) -> List[tuple]:
        if not self.vocabulary:
            return []
        
        words = list(self.vocabulary.keys())[:top_n]
        return [(word, idx) for idx, word in enumerate(words)]


def create_text_vectors(data: List[Dict[str, str]], 
                       method: str = 'tfidf',
                       use_lemmatization: bool = True) -> Dict[str, np.ndarray]:

    vectorizer = TextVectorizer(use_lemmatization=use_lemmatization)
    
    if method == 'tf':
        vectors = vectorizer.vectorize_texts(data)
    elif method == 'tfidf':
        vectors = vectorizer.vectorize_with_tfidf(data)
    else:
        raise ValueError(f"Неизвестный метод: {method}. Используйте 'tf' или 'tfidf'.")
    
    return vectors


if __name__ == "__main__":
    sample_data = [
        {
            'file': 'doc1.txt',
            'text': 'Машинное обучение — это область искусственного интеллекта.'
        },
        {
            'file': 'doc2.txt', 
            'text': 'Глубокое обучение является подразделом машинного обучения.'
        },
        {
            'file': 'doc3.txt',
            'text': 'Искусственный интеллект и машинное обучение активно развиваются.'
        }
    ]
    
    print("Векторизация с лемматизацией (TF-IDF):")
    vectors_lemmatized = create_text_vectors(sample_data, method='tfidf', use_lemmatization=True)
    
    for filename, vector in vectors_lemmatized.items():
        print(f"\n{filename}:")
        print(f"  Размер вектора: {len(vector)}")
        print(f"  Ненулевые элементы: {np.sum(vector > 0)}")
        print(f"  Пример первых 10 значений: {vector[:10]}")
    
    print("\n" + "="*50)
    print("Векторизация без лемматизации (словоформы, TF):")
    vectors_wordforms = create_text_vectors(sample_data, method='tf', use_lemmatization=False)
    
    for filename, vector in vectors_wordforms.items():
        print(f"\n{filename}:")
        print(f"  Размер вектора: {len(vector)}")
        print(f"  Ненулевые элементы: {np.sum(vector > 0)}")
    
    print("\n" + "="*50)
    print("Пример работы лемматизатора:")
    vectorizer = TextVectorizer(use_lemmatization=True)
    sample_text = "Коты бежали по дороге и увидели собак"
    lemmatized = vectorizer.preprocess_text(sample_text)
    print(f"Исходный текст: {sample_text}")
    print(f"После лемматизации: {lemmatized}")
    
    vectorizer_no_lemma = TextVectorizer(use_lemmatization=False)
    wordforms = vectorizer_no_lemma.preprocess_text(sample_text)
    print(f"Словоформы: {wordforms}")