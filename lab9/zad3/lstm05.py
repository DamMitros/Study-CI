# Load LSTM network and generate text
import sys
import numpy as np
from nltk.tokenize import wordpunct_tokenize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
# load ascii text and covert to lowercase
filename = "wonderland.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
# create mapping of unique chars to integers
tokenized_text = wordpunct_tokenize(raw_text)
tokens = sorted(list(dict.fromkeys(tokenized_text)))

#print("Tokens: ")
#print(tokens)
tok_to_int = dict((c, i) for i, c in enumerate(tokens))
int_to_tok = dict((i, c) for i, c in enumerate(tokens))
#print("TokensToNumbers: ")
#print(tok_to_int)

# summarize the loaded data
n_tokens = len(tokenized_text)
n_token_vocab = len(tokens)
print("Total Tokens: ", n_tokens)
print("Unique Tokens (Token Vocab): ", n_token_vocab)

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_tokens - seq_length, 1):
	seq_in = tokenized_text[i:i + seq_length]
	seq_out = tokenized_text[i + seq_length]
	dataX.append([tok_to_int[tok] for tok in seq_in])
	dataY.append(tok_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_token_vocab)
# one hot encode the output variable
y = to_categorical(dataY)
# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
# load the network weights
filename = "big-token-model-2-30-3.1710.keras"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')
# pick a random seed
start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print("Seed:")
print("\"", ' '.join([int_to_tok[value] for value in pattern]), "\"")
# generate tokens
print("Generated text:")
for i in range(100):
	x = np.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_token_vocab)
	prediction = model.predict(x, verbose=0)
	index = np.argmax(prediction)
	result = int_to_tok[index]
	seq_in = [int_to_tok[value] for value in pattern]
	sys.stdout.write(result+" ")
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
print("\nDone.")


# Program:
#    1. Wczytanie danych
#	 2. Przygotowanie danych do uczenia
#	 3. Zdefiniowanie modelu LSTM
#	 4. Uczenie modelu (wczytanie wag)
#	 5. Generowanie tekstu

# Program różni się od poprzedniego tym, że zamiast całych słów używa znaków do generowania tekstu.

# Wygenerowany tekst (po pierwszym uruchomieniu, 20 epok):
# the the , “ the , , “ i ’ s ’ t ,” said the , said ’ t , “ the , you t t you , said the , “ ’ t , 
# “ the t you , said the , “ the t , “ the , you t t you , said the , said ’ t , “ the , you t t you ,
#  said the , said ’ t , “ the , you t t you , said the , said ’ t , “ the , you t t you

# Wygenerowany tekst (po doszkoleniu, 50 epok):
# remark . “ yes ’ s none ?” said the hatter , “ i ’ ’ t classics to names , “ i ’ s ’ t a the the ,” 
# said the caterpillar hare , “ the king , “ i ’ m a said the , “ , “ the , “ the t it , “ you , won ’ t you ,
# will you t you join will dance , “ you , won ’ t you t will mock , “ you , won ’ t you join will dance doth the you busy it 

# Wygerowany teskt po doszkoleniu działa dużo lepiej, ale wciąż nie jest idealny. Słowa mają sens, ale nie są poprawnie ułożone w zdania.
# W porównaniu do poprzedniego modelu, ten model generuje tekst bardziej realistyczny, ale w krótkich fragmentach.