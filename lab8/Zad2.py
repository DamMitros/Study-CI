import matplotlib.pyplot as plt
import nltk
import text2emotion as te
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

positive_review = """
The best travel experience we’ve had in a long time!
Agriturismo Humile was a little gem I’m so thrilled we were able to discover. Tommaso (the host) was warm and welcoming in every single interaction we had, ensuring we were comfortable and had what we needed for our stay.

The family who owns this property has put so much love and heart into it and truly want their guests to have an unforgettable experience, and they have absolutely succeeded.

The homemade breakfast, the olive harvesting happening before our eyes, their sweet dogs running around - everywhere you looked, you saw joy and beauty.

We will absolutely be back!
"""

negative_review = """
Terrible place for families and even worse for couples who just want to 'Glamp' and feel at ease...
Staff was consistently rude and unhelpful, and the glamping experience felt like a cheap, overcrowded tourist trap: A lot of (fellow) Dutch people, who obnoxiously leave their towels on the beds at the swimming pool - an infuriating and inconsiderate habit...
The restaurant was outrageously expensive with awful food for dinner, barely tolerable snacks for lunch.
The place was severely lacking in beach chairs - to get one, we had to endure a humiliating queue by the swimming pool entrance before opening hours. The entire swimming pool area was chaotic, overcrowded, and utterly unpleasant.
"""

# 2) Analiza sentymentu używając VADER
print("Step 2: Sentiment Analysis using VADER")
sia = SentimentIntensityAnalyzer()

positive_scores = sia.polarity_scores(positive_review)
print("\nPositive review:") 
print(f"Positive: {positive_scores['pos']:.3f}")
print(f"Neutral: {positive_scores['neu']:.3f}")
print(f"Negative: {positive_scores['neg']:.3f}")
print(f"Compound: {positive_scores['compound']:.3f}")

negative_scores = sia.polarity_scores(negative_review)
print("\nNegative review:")
print(f"Positive: {negative_scores['pos']:.3f}")
print(f"Neutral: {negative_scores['neu']:.3f}")
print(f"Negative: {negative_scores['neg']:.3f}")
print(f"Compound: {negative_scores['compound']:.3f}")

# 3) Analiza emocji używając Text2Emotion
print("\nStep 3: Emotion Analysis using Text2Emotion")

def print_text2emotion_results(review_type, emotions):
    print(f"\n{review_type} (Emotions - Text2Emotion):")
    if emotions:
        for emotion, score in emotions.items():
            print(f"  - {emotion}: {score:.3f}")
        dominant_emotion = max(emotions, key=emotions.get)
        print(f"  Dominant Emotion: {dominant_emotion}")
    else:
        print("Could not retrieve emotion results.")

t2e_positive_result = te.get_emotion(positive_review)
print_text2emotion_results("Positive review", t2e_positive_result)

t2e_negative_result = te.get_emotion(negative_review)
print_text2emotion_results("Negative review", t2e_negative_result)

# 4) Wyniki

# Step 2: Sentiment Analysis using VADER

# Positive review:
# Positive: 0.309
# Neutral: 0.691
# Negative: 0.000
# Compound: 0.991

# Negative review:
# Positive: 0.028
# Neutral: 0.679
# Negative: 0.293
# Compound: -0.987

# Step 3: Emotion Analysis using Text2Emotion

# Positive review (Emotions - Text2Emotion):
#   - Happy: 0.290
#   - Angry: 0.070
#   - Surprise: 0.210
#   - Sad: 0.140
#   - Fear: 0.290
#   Dominant Emotion: Happy

# Negative review (Emotions - Text2Emotion):
#   - Happy: 0.050
#   - Angry: 0.000
#   - Surprise: 0.160
#   - Sad: 0.260
#   - Fear: 0.530
#   Dominant Emotion: Fear

# Vader poradził sobie świetnie z rozpoznanie pozytywnej i negatywnej recenzji.
# Z kolei Text2Emotion nie poradził sobie najlepiej, w obu przypadkach wykrył
# dominującą emocję jako strach, gdzie nie wydaje się to być prawdą.
# Oczekiwałbym bardziej w pozytywnej recenzji emocji szczęścia, a w negatywnej
# emocji gniewu i smutku.