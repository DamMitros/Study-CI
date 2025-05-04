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