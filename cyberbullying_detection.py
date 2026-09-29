import pandas as pd
import re
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Download required NLTK data
nltk.download("stopwords")
nltk.download("wordnet")

# Step 1: Create or load your dataset
data = pd.DataFrame({
    "text": [
        "You are so stupid and worthless",
        "I hope you die",
        "Have a great day!",
        "Your work is awesome",
        "Fuck you idiot",
        "You are such a bitch",
        "You are looking beautiful",
        "You are so ugly",
        "Stop talking, clown",
        "You are a waste of space",
        "I hate you so much",
        "You are too dumb to understand",
        "No one wants you here",
        "You are very talented",
        "Thank you for your support",
        "I like your positive attitude",
        "You inspire me",
        "You are improving every day",
        "That was a great presentation",
        "You are a good friend",
        "You look fantastic today",
        # Add more examples as needed.
    ],
    # 1 = cyberbullying, 0 = normal
    "label": [
        1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0
    ],
})

# Step 2: Text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


data["text"] = data["text"].apply(preprocess_text)

# Step 3: Split data into training and testing sets
X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# Step 4: TF-IDF + Naive Bayes pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=1000, stop_words="english")),
    ("classifier", MultinomialNB()),
])

# Step 5: Train the model
pipeline.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = pipeline.predict(X_test)

# Step 7: Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1-Score:", f1_score(y_test, y_pred, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Step 8: Test new text
def detect_cyberbullying(text):
    processed_text = preprocess_text(text)
    prediction = pipeline.predict([processed_text])[0]
    confidence = pipeline.predict_proba([processed_text]).max()

    if prediction == 1:
        return f"CYBERBULLYING DETECTED (Confidence: {confidence:.2%})"
    return f"Normal text (Confidence: {confidence:.2%})"


test_messages = [
    "You are such a loser",
    "This is a nice day",
    "i will fuck you",
    "im going home",
    "you are mad",
    "you are number 1 idiot",
    "I like your shirt",
    "you are looking good today!",
    "I am sorry if I upset you",
    "I am going to beat you up after school",
    "you are looking so dirty",
    "she is so dumb to handle",
    "what the freak",
    "he is very bad",
    "they are fucking idiots",
    "you are so kind",
    "everyone likes you",
]

for message in test_messages:
    print(f"{message!r} -> {detect_cyberbullying(message)}")
