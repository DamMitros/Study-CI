import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def forwardPass(wiek, waga, wzrost):
    w1, w2, w3 = -0.46122, 0.97314, -0.39203 
    w4, w5, w6 = 0.78548, 2.10584, -0.57847
    bias1, bias2 = 0.80109, 0.43529
    w_out1, w_out2 = -0.81546, 1.03775
    bias_out = -0.2368

    hidden1 = (w1 * wiek) + (w2 * waga) + (w3 * wzrost) + bias1
    hidden1_po_aktywacji = sigmoid(hidden1)
    hidden2 = (w4 * wiek) + (w5 * waga) + (w6 * wzrost) + bias2
    hidden2_po_aktywacji = sigmoid(hidden2)
    output = (w_out1 * hidden1_po_aktywacji) + (w_out2 * hidden2_po_aktywacji) + bias_out

    return output

data = [
    [23, 75, 176],
    [25, 67, 180],
    [28, 120, 175],
    [22, 65, 165],
    [46, 70, 187],
    [50, 68, 180],
    [48, 97, 178] 
]

def main():
    total_output = 0
    for osoba in data:
        wiek, waga, wzrost = osoba
        wynik = forwardPass(wiek, waga, wzrost)
        if wynik > 0.5:
            print(f"Osoba {osoba} będzie grała w siatkówkę")
        else:
            print(f"Osoba {osoba} nie będzie grała w siatkówkę")
        total_output += wynik
        print(f"Osoba {osoba}: wynik = {wynik}")

main()