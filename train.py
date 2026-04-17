import pandas as pd
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords', quiet=True)

# loading the dataset
df = pd.read_csv('data/spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

ham_count = len(df[df['label'] == 'ham'])
spam_count = len(df[df['label'] == 'spam'])
print(f"Loaded {len(df)} messages ({ham_count} ham, {spam_count} spam)")

# cleaning the text
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove numbers and special chars
    words = text.split()

    cleaned = []
    for word in words:
        if word not in stop_words:
            stemmed = stemmer.stem(word)
            cleaned.append(stemmed)

    return ' '.join(cleaned)

print("Cleaning text...")
df['cleaned'] = df['message'].apply(clean_text)

# converting text to tfidf vectors
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['cleaned'])

# converting labels to 0 and 1
y = df['label'].map({'ham': 0, 'spam': 1})

# splitting into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# training naive bayes
print("Training model...")
model = MultinomialNB()
model.fit(X_train, y_train)

# checking accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# saving model and vectorizer
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("Model saved, run predict.py to test")