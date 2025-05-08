# Load LSTM network and generate text
import sys
import numpy as np
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
# create mapping of unique chars to integers, and a reverse mapping
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)
# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = to_categorical(dataY)
# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
# load the network weights
filename = "wagi_lstm02/weights-improvement-2-23-1.6646.keras"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')
# pick a random seed
start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print("Seed:")
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
# generate characters
for i in range(500):
	x = np.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = np.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
print("\nDone.")

# Program:
#    1. Wczytanie danych 
#	 2. Przygotowanie danych do uczenia
#	 3. Zdefiniowanie modelu LSTM
#	 4. Uczenie modelu (wczytanie wag)
#	 5. Generowanie tekstu

# Po pierwszym uruchomieniu (50 epok):
	# Seed:
	# " ch a capital one for catching mice—oh, i beg your
	# pardon!” cried alice again, for this time the mous "

	# Wygenerowany teskt:
	# e oo the tooe of the taale, aut she was not io the toice of the taate, 
	# “the was io the carte nitt ar a lott,” said alice, “io you think yhu moge the dante. 
	# “h woi’t!in the wert or trint,” the macc to see whet wo beard. “he you don’t know that toen would be a hadd of the coor—!
	#                                  *l waat wou denee ee mouee o
	# thou woul 

# Do doszkoleniu (50 + 23 epok)
	# Seed:
	# " , we went to school every day—”
	# “_i’ve_ been to a day-school, too,” said alice; “you needn’t be so 

	# Wygenerowany tekst:
	# anoiers, to ce anlie, i wonler in mot anate  who ind toe taid to toeng io the tei! she was as the lictter food ano thet wiue tie lade of the cand. 
	# “here your eanlen fateer in the sine,” she said this, an see sookd the dat and she was thei it har and soenengd at the oook of the taate, 
	# “thel i sas aoinn_ in the sial the garc to toit,” the gatter weit on, “in you drolt done the doort  a dat  io your or daan ii i can’t belte th 
	# seyed and iore ano the cronons, a dorio and aroent thanle at the oadt

# Wygenerowany tekst jest nieco bardziej sensowny, ale wciąż nie jest to idealny wynik. Najbardziej zauważalna jest próba 
# generowania dłuższych słów, dzięki czemu tekst wydaje się bardziej spójny i dłuższy.