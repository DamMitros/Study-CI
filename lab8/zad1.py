import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

file_path = 'article.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

print("Original text sample (first 20 characters):")
print(text[:20], "...\n")

# 2) Tokenizacja
tokens = word_tokenize(text)
print(f"Step 2: Number of tokens after tokenization: {len(tokens)}")
print(f"Sample tokens: {tokens[:10]}\n")

# 3) Usunięcie stop-words
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]
print(f"Step 3: Number of tokens after removing stopwords: {len(filtered_tokens)}")
print(f"Sample tokens after stopword removal: {filtered_tokens[:20]}\n")

# 4) Stop-words 
stop_words.update(['said'])
filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]
print(f"Step 4: Number of tokens after updating stopwords list: {len(filtered_tokens)}")
# print(f"Number of stopwords: {len(stop_words)}\n")
# print(stop_words)

# 5) Lematyzacjaja
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word.lower()) for word in filtered_tokens]
print(f"\nStep 5: Number of tokens after lemmatization: {len(lemmatized_tokens)}")
print(f"Lemmatizer used: WordNetLemmatizer")
print(f"Sample lemmatized tokens: {lemmatized_tokens[:20]}\n")

# WordNetLemmatizer został użyty do lematyzacji, ze względu łatwośc integracji z NLTK i wykorzystywanie WordNet jako źródła danych.

# 6) Zliczanie częstotliwości słów +wykres
word_freq = Counter(lemmatized_tokens)
top_words = word_freq.most_common(10)
print(f"Step 6: Top 10 most frequent words:")
for word, count in top_words:
  print(f"  {word}: {count}")

words, counts = zip(*top_words)
plt.figure(figsize=(12, 6))
plt.bar(words, counts)
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 10 Most Frequent Words')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('top_words.png')

# 7) Chmurka słów
wordcloud = WordCloud(width=800, height=400, 
                     background_color='white', 
                     max_words=100, 
                     contour_width=3, 
                     contour_color='steelblue').generate(' '.join(lemmatized_tokens))

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout()
plt.savefig('wordcloud.png')
print("Word cloud saved as 'wordcloud.png'")
