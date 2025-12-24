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
        """
        Инициализация векторизатора текста
        
        Args:
            use_lemmatization: Использовать лемматизацию (True) или словоформы (False)
            min_word_length: Минимальная длина слова для включения в вектор
            remove_stopwords: Удалять стоп-слова
            custom_stopwords: Пользовательский список стоп-слов
        """
        self.use_lemmatization = use_lemmatization
        self.min_word_length = min_word_length
        self.remove_stopwords = remove_stopwords
        
        # Инициализация морфологического анализатора
        self.morph_analyzer = pymorphy3.MorphAnalyzer()
        
        # Список стоп-слов
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
        
        # Словарь для хранения словаря всех слов
        self.vocabulary = {}
        self.vocabulary_size = 0
        
    def preprocess_text(self, text: str) -> List[str]:
        """
        Предварительная обработка текста
        
        Args:
            text: Исходный текст
            
        Returns:
            Список обработанных слов
        """
        # Приведение к нижнему регистру
        text = text.lower()
        
        # Удаление символов, кроме букв и пробелов
        text = re.sub(r'[^а-яёa-z\s]', ' ', text)
        
        # Разделение на слова
        words = text.split()
        
        # Фильтрация и нормализация слов
        processed_words = []
        for word in words:
            # Пропуск коротких слов
            if len(word) < self.min_word_length:
                continue
                
            # Удаление стоп-слов
            if self.remove_stopwords and word in self.stopwords:
                continue
            
            # Применение морфологического анализа
            if self.use_lemmatization:
                # Лемматизация - приведение к начальной форме
                parsed_word = self.morph_analyzer.parse(word)[0]
                normalized_word = parsed_word.normal_form
            else:
                # Использование словоформ (оставляем как есть, после очистки)
                normalized_word = word
            
            processed_words.append(normalized_word)
        
        return processed_words
    
    def build_vocabulary(self, texts: List[str]) -> None:
        """
        Построение словаря на основе всех текстов
        
        Args:
            texts: Список текстов для обучения
        """
        all_words = []
        
        for text in texts:
            words = self.preprocess_text(text)
            all_words.extend(words)
        
        # Подсчет частоты слов
        word_counts = Counter(all_words)
        
        # Создание словаря с индексами
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(
            sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        )}
        self.vocabulary_size = len(self.vocabulary)
    
    def vectorize_text(self, text: str) -> np.ndarray:
        """
        Векторизация отдельного текста
        
        Args:
            text: Текст для векторизации
            
        Returns:
            Вектор текста
        """
        if not self.vocabulary:
            raise ValueError("Словарь не построен. Сначала вызовите build_vocabulary.")
        
        words = self.preprocess_text(text)
        
        # Создание вектора
        vector = np.zeros(self.vocabulary_size, dtype=np.float32)
        
        # Подсчет частот слов
        for word in words:
            if word in self.vocabulary:
                idx = self.vocabulary[word]
                vector[idx] += 1
        
        # Нормализация вектора (TF)
        if np.sum(vector) > 0:
            vector = vector / np.sum(vector)
        
        return vector
    
    def vectorize_texts(self, data: List[Dict[str, str]]) -> Dict[str, np.ndarray]:
        """
        Векторизация списка текстов
        
        Args:
            data: Список словарей с ключами 'file' и 'text'
            
        Returns:
            Словарь с векторами для каждого файла
        """
        # Извлечение всех текстов
        texts = [item['text'] for item in data]
        
        # Построение словаря
        self.build_vocabulary(texts)
        
        # Векторизация каждого текста
        vectors = {}
        for item in data:
            filename = item['file']
            text = item['text']
            vector = self.vectorize_text(text)
            vectors[filename] = vector
        
        return vectors
    
    def vectorize_with_tfidf(self, data: List[Dict[str, str]]) -> Dict[str, np.ndarray]:
        """
        Векторизация с использованием TF-IDF
        
        Args:
            data: Список словарей с ключами 'file' и 'text'
            
        Returns:
            Словарь с TF-IDF векторами для каждого файла
        """
        # Извлечение всех текстов
        texts = [item['text'] for item in data]
        
        # Построение словаря
        self.build_vocabulary(texts)
        
        # Подсчет частот документов (DF)
        doc_freq = np.zeros(self.vocabulary_size, dtype=np.float32)
        
        for item in data:
            words = set(self.preprocess_text(item['text']))
            for word in words:
                if word in self.vocabulary:
                    idx = self.vocabulary[word]
                    doc_freq[idx] += 1
        
        # Вычисление IDF
        num_docs = len(data)
        idf = np.log((num_docs + 1) / (doc_freq + 1)) + 1
        
        # Векторизация каждого текста с TF-IDF
        vectors = {}
        for item in data:
            filename = item['file']
            text = item['text']
            
            # TF часть
            tf_vector = np.zeros(self.vocabulary_size, dtype=np.float32)
            words = self.preprocess_text(text)
            
            for word in words:
                if word in self.vocabulary:
                    idx = self.vocabulary[word]
                    tf_vector[idx] += 1
            
            # Нормализация TF
            if np.sum(tf_vector) > 0:
                tf_vector = tf_vector / np.sum(tf_vector)
            
            # TF-IDF
            tfidf_vector = tf_vector * idf
            
            # Нормализация L2
            norm = np.linalg.norm(tfidf_vector)
            if norm > 0:
                tfidf_vector = tfidf_vector / norm
            
            vectors[filename] = tfidf_vector
        
        return vectors
    
    def get_most_frequent_words(self, top_n: int = 20) -> List[tuple]:
        """
        Получение самых частых слов в словаре
        
        Args:
            top_n: Количество возвращаемых слов
            
        Returns:
            Список кортежей (слово, частота)
        """
        if not self.vocabulary:
            return []
        
        # Для простоты возвращаем топ-N слов по индексу (словарь построен по убыванию частоты)
        words = list(self.vocabulary.keys())[:top_n]
        return [(word, idx) for idx, word in enumerate(words)]


# Пример использования функции
def create_text_vectors(data: List[Dict[str, str]], 
                       method: str = 'tfidf',
                       use_lemmatization: bool = True) -> Dict[str, np.ndarray]:
    """
    Основная функция для создания векторов текстов
    
    Args:
        data: Список словарей [{'file': имя_файла, 'text': текст}, ...]
        method: Метод векторизации ('tf', 'tfidf')
        use_lemmatization: Использовать лемматизацию
        
    Returns:
        Словарь с векторами для каждого файла
    """
    # Инициализация векторизатора
    vectorizer = TextVectorizer(use_lemmatization=use_lemmatization)
    
    # Векторизация в зависимости от метода
    if method == 'tf':
        vectors = vectorizer.vectorize_texts(data)
    elif method == 'tfidf':
        vectors = vectorizer.vectorize_with_tfidf(data)
    else:
        raise ValueError(f"Неизвестный метод: {method}. Используйте 'tf' или 'tfidf'.")
    
    return vectors


# Пример использования
if __name__ == "__main__":
    # Пример данных
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
    
    # Векторизация с лемматизацией (начальные формы слов)
    print("Векторизация с лемматизацией (TF-IDF):")
    vectors_lemmatized = create_text_vectors(sample_data, method='tfidf', use_lemmatization=True)
    
    for filename, vector in vectors_lemmatized.items():
        print(f"\n{filename}:")
        print(f"  Размер вектора: {len(vector)}")
        print(f"  Ненулевые элементы: {np.sum(vector > 0)}")
        print(f"  Пример первых 10 значений: {vector[:10]}")
    
    # Векторизация без лемматизации (словоформы)
    print("\n" + "="*50)
    print("Векторизация без лемматизации (словоформы, TF):")
    vectors_wordforms = create_text_vectors(sample_data, method='tf', use_lemmatization=False)
    
    for filename, vector in vectors_wordforms.items():
        print(f"\n{filename}:")
        print(f"  Размер вектора: {len(vector)}")
        print(f"  Ненулевые элементы: {np.sum(vector > 0)}")
    
    # Демонстрация работы лемматизатора
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