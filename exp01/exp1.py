import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

text = input("Enter a sentence: ")

stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in text]

lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in text]

print("\nOriginal Text:")
print(text)
print("\nTokens:")
print(tokens)
print("\nStemmed Words:")
print(stemmed_words)
print("\nLemmatized Words:")
print(lemmatized_words)