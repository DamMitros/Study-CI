import numpy as np
from keras.layers import SimpleRNN

inputs = np.random.random([32, 10, 8]).astype(np.float32) # losowe dane wejściowe (32 przykłady, 10 kroków czasowych, 8 cech)
print("Inputs: ")
print(inputs)

simple_rnn = SimpleRNN(4) # Podstawowy RNN z 4 jednostkami ukrytymi (neuronami)

output = simple_rnn(inputs)  # The output has shape `[32, 4]`.
print("Output: ")
print(output)

simple_rnn = SimpleRNN(
    4, return_sequences=True, return_state=True) # RNN z 4 jednostkami ukrytymi, zwracający każdą sekwencje i stan końcowy

# whole_sequence_output has shape `[32, 10, 4]`.
# final_state has shape `[32, 4]`.
whole_sequence_output, final_state = simple_rnn(inputs)

# print("Whole sequence output: ")
# print(whole_sequence_output)

# print("Final state: ")
# print(final_state)

# Program:
#   1. Tworzy losowe dane wejściowe o kształcie `[32, 10, 8]` (32 przykłady, 10 kroków czasowych, 8 cech).
#   2. Tworzy prosty model RNN z 4 jednostkami ukrytymi.
#   3. Przetwarza dane wejściowe przez model RNN, uzyskując wyjście o kształcie `[32, 4]`.
#   4. Tworzy model RNN z 4 jednostkami ukrytymi, który zwraca pełną sekwencję i stan końcowy.

#   Pierwszy model zwraca tylko ostatnie wyjście, podczas gdy drugi model zwraca wszystkie wyjścia oraz stan końcowy.
#   Oba modele przetwarzają dane wejściowe o tym samym kształcie, ale różnią się sposobem zwracania wyników.