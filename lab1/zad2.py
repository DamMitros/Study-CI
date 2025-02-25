import random, math, matplotlib.pyplot as plt

def shootedplace(alpha, target):
  v=50; h=100; g=9.81
  alpha = math.radians(alpha)

  d=(v*math.cos(alpha)) * ((v*math.sin(alpha)+math.sqrt((v*math.sin(alpha))**2+2*g*h))/g)

  if target-5 <= d <= target+5:
    plot(alpha, v, h, g)
    return True
  else:
    print(f"Strzał oddano na odległość {d:.2f} metrów.")
    return False
  
def plot(alpha, v, h, g):
  x=[0]; y=[h]; t=0; dt=0.1
  while True:
    t += dt
    x.append(v*t*math.cos(alpha))
    y.append(h + v*t*math.sin(alpha) - 0.5*g*t**2)

    if y[-1] < 0:
      break

  plt.plot(x, y)
  plt.xlabel("Dystans [m]")
  plt.ylabel("Wysokość [m]")
  plt.title("Trajektoria pocisku")
  plt.grid(True)
  plt.savefig("trajektoria.png")
 
def main():
  attempts = 0
  target = random.randint(50,340)
  print(f"Cel znajduje się na odległości {target} metrów.\n")

  while True:
    attempts += 1
    alpha = int(input("Podaj kąt strzału: "))
    if shootedplace(alpha, target):
      print("Cel trafiony!\n")
      print(f"Gratulacje! Łączna liczba prób: {attempts}")
      break
    else:
      print("Pudło. Spróbuj ponownie.\n")

main()