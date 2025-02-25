import datetime, math

def calculate_biorhythm(days, period):
  """Calculates the biorhythm value for a given period."""
  return math.sin(2 * math.pi * days / period)

def check_happiness(value, name, days):
  """Checks the biorhythm value and provides feedback."""
  if value > 0.5:
    print(f"Your {name} biorhythm is high! It's going to be a good day!")
  elif value < -0.5:
    print(f"Your {name} biorhythm is low.")
    tomorrow = days + 1
    next_value = calculate_biorhythm(tomorrow, 23 if name == "fizyczny" else 28 if name == "emocjonalny" else 33)
    if next_value > value:
      print("Don't worry! Tomorrow will be better!")
    else:
      print("The next day might be similar.")

def get_user_input():
  """Gets user input for name and birthdate."""
  name = input("Enter your name: ")
  year = int(input("Enter your birth year: "))
  month = int(input("Enter your birth month: "))
  day = int(input("Enter your birth day: "))
  return name, year, month, day

def cycle():
  """Calculates and displays biorhythms."""
  name, year, month, day = get_user_input()

  current_date = datetime.datetime.now()
  birth_date = datetime.datetime(year, month, day)
  days_lived = (current_date - birth_date).days

  physical_biorhythm = calculate_biorhythm(days_lived, 23)
  emotional_biorhythm = calculate_biorhythm(days_lived, 28)
  intellectual_biorhythm = calculate_biorhythm(days_lived, 33)

  print(f"Hello, {name}! You have lived for {days_lived} days!")

  print(f"Your physical biorhythm is: {physical_biorhythm:.2f}")
  check_happiness(physical_biorhythm, "physical", days_lived)

  print(f"Your emotional biorhythm is: {emotional_biorhythm:.2f}")
  check_happiness(emotional_biorhythm, "emotional", days_lived)

  print(f"Your intellectual biorhythm is: {intellectual_biorhythm:.2f}")
  check_happiness(intellectual_biorhythm, "intellectual", days_lived)

cycle()

# Poprosiłem Gemini aby wystylizował i poprawił kod. Zmiany przede wszystkim polegały na większym podziale
# kodu na funkcje, dodaniu komentarzy, zmianie nazw zmiennych dodaniu formatowania do wyświetlanych wartości