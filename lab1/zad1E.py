import datetime
import math

def get_biorhythm(days, cycle):
    return math.sin(2 * math.pi * days / cycle)

def main():
    # Pobranie danych od użytkownika
    name = input("Podaj swoje imię: ")
    year = int(input("Podaj rok urodzenia: "))
    month = int(input("Podaj miesiąc urodzenia: "))
    day = int(input("Podaj dzień urodzenia: "))
    
    birth_date = datetime.date(year, month, day)
    today = datetime.date.today()
    
    days_lived = (today - birth_date).days
    print(f"\nCześć, {name}! Dzisiaj jest {today}. To Twój {days_lived} dzień życia.\n")
    
    # Obliczenie biorytmów
    physical = get_biorhythm(days_lived, 23)
    emotional = get_biorhythm(days_lived, 28)
    intellectual = get_biorhythm(days_lived, 33)
    
    print(f"Twoje biorytmy na dziś:")
    print(f"Fizyczny: {physical:.2f}")
    print(f"Emocjonalny: {emotional:.2f}")
    print(f"Intelektualny: {intellectual:.2f}\n")
    
    # Obliczenie biorytmów na jutro
    physical_tomorrow = get_biorhythm(days_lived + 1, 23)
    emotional_tomorrow = get_biorhythm(days_lived + 1, 28)
    intellectual_tomorrow = get_biorhythm(days_lived + 1, 33)
    
    # Analiza wyników
    for category, value, value_tomorrow in zip(
        ["Fizyczny", "Emocjonalny", "Intelektualny"],
        [physical, emotional, intellectual],
        [physical_tomorrow, emotional_tomorrow, intellectual_tomorrow]
    ):
        if value > 0.5:
            print(f"Gratulacje! Twój {category.lower()} poziom jest wysoki ({value:.2f}). To dobry dzień!")
        elif value < -0.5:
            print(f"Twój {category.lower()} poziom jest niski ({value:.2f}). To może być trudniejszy dzień.")
            if value_tomorrow > value:
                print("Nie martw się. Jutro będzie lepiej!")
    
if __name__ == "__main__":
    main()

# Poprosiłem ChatGPT o wykonanie kodu. Zrobił to błyskawicznie, w bardzo podobny sposób. Przede wszystkim główną 
# różnicą jest to, że bot zastosował pętlę przy analizie wyników oraz uniwersalną funckję do obliczania biorytmów.