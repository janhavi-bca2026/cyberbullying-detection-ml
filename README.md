# Cyberbullying Detection Using Machine Learning

A simple Natural Language Processing (NLP) project that classifies text as
**cyberbullying** or **normal text** using TF-IDF features and a Multinomial
Naive Bayes classifier.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- TF-IDF
- Multinomial Naive Bayes

## How It Works

1. Text is converted to lowercase.
2. Special characters and extra whitespace are removed.
3. The data is split into training and testing sets.
4. TF-IDF converts text into numerical features.
5. Multinomial Naive Bayes trains the classifier.
6. Accuracy, precision, recall, F1-score, and a confusion matrix are displayed.
7. New text can be passed to `detect_cyberbullying()` for classification.

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python cyberbullying_detection.py
```

The first run downloads the required NLTK resources.

## Project Note

The included dataset is a small demonstration dataset created for this
academic project. The model should not be treated as a production-level
cyberbullying detector. A larger, diverse, and properly validated dataset
would be needed for reliable real-world use.
