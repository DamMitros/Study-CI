import datetime, math

def cycle():
  name = input("Podaj swoje imię: ")
  year = int(input("Podaj swój rok urodzenia: "))
  month = int(input("Podaj swój miesiąc urodzenia: "))
  day = int(input("Podaj swój dzień urodzenia: "))

  currentdate = datetime.datetime.now()
  birthdate = datetime.datetime(year, month, day)
  dayofyourlife = currentdate - birthdate

  physicalfaze = math.sin(2 * math.pi * dayofyourlife.days / 23) 
  emotionalfaze = math.sin(2 * math.pi * dayofyourlife.days / 28)
  intellectualfaze = math.sin(2 * math.pi * dayofyourlife.days / 33) 

  print(f"Witaj, {name}! Żyjesz już {dayofyourlife.days} dni!")
  print(f"Twoja fizyczna faza życia to: {physicalfaze}")
  check_hapiness(physicalfaze, "fizyczny", dayofyourlife.days)
  print(f"Twoja emocjonalna faza życia to: {emotionalfaze}")
  check_hapiness(emotionalfaze, "emocjonalny", dayofyourlife.days)
  print(f"Twoja intelektualna faza życia to: {intellectualfaze}")
  check_hapiness(intellectualfaze, "intelektualny", dayofyourlife.days)

def check_hapiness(value, name, days):
  if value > 0.5:
    print(f'Twój {name} biorytm jest wysoki! To będzie dobry dzień!')
  elif value < -0.5:
    tomorrow = days + 1
    next_value = 0
    
    if name == "fizyczny":
      next_value = math.sin(2 * math.pi * tomorrow / 23)
    elif name == "emocjonalny":
      next_value = math.sin(2 * math.pi * tomorrow / 28)
    elif name == "intelektualny":
      next_value = math.sin(2 * math.pi * tomorrow / 33)
      
    print(f'Twój {name} biorytm jest niski.')
    if next_value > value:
      print(f'Nie martw się! Jutro będzie lepiej!')
    else:
      print(f'Następny dzień może być podobny.')

cycle()

# Całość zajęła mi około 30 minut. Najwięcej czasu zajęło mi zrozumienie działania funkcji check_hapiness.

