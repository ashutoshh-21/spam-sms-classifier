import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)

# loading the trained model and tfidf vectorizer
model = pickle.load(open('models/model.pkl', 'rb'))
tfidf = pickle.load(open('models/vectorizer.pkl', 'rb'))

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# this function cleans the input text same way we did during training
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # removing numbers and special chars
    words = text.split()
    cleaned = []
    for word in words:
        if word not in stop_words:
            cleaned.append(stemmer.stem(word))
    return ' '.join(cleaned)


def predict(msg):
    msg = clean_text(msg)
    msg_vector = tfidf.transform([msg])
    result = model.predict(msg_vector)[0]
    proba = model.predict_proba(msg_vector)[0]
    confidence = max(proba) * 100
    return result, confidence


# main
print("\nSMS Spam Detector")
print("Enter a message to check if it is spam or not")
print("Type 'quit' to exit\n")

while True:
    msg = input("Enter message: ")
    msg = msg.strip()

    if msg.lower() == 'quit':
        break

    if msg == "":
        print("Enter something first\n")
        continue

    result, confidence = predict(msg)

    if result == 1:
        print("Result: SPAM (confidence: {:.1f}%)\n".format(confidence))
    else:
        print("Result: NOT SPAM (confidence: {:.1f}%)\n".format(confidence))