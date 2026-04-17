# README.md


# SMS Spam Detector

A simple command-line tool that detects whether an SMS message is spam or not.
Built using Python and basic machine learning techniques as part of my
Fundamentals of AI/ML course project.

## What does it do?

You type a message in the terminal. The program tells you if it's spam or
not spam, along with how confident it is about the prediction.

```
Enter message: Congratulations! You won a free iPhone! Claim now!
>> SPAM (confidence: 92.9%)

Enter message: Hey are you free for lunch today?
>> NOT SPAM (confidence: 97.3%)
```

## How it works

1. The model is trained on the [UCI SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
   which contains around 5,500 real SMS messages labeled as spam or ham (not spam).

2. The text goes through a cleaning process — lowercase conversion,
   removing punctuation and numbers, removing common filler words (stopwords),
   and reducing words to their root form (stemming).

3. Cleaned text is converted to numbers using TF-IDF
   (Term Frequency - Inverse Document Frequency), which basically scores
   each word based on how important it is.

4. A Naive Bayes classifier is trained on these numbers to learn
   patterns that separate spam from normal messages.

5. When you type a new message, it goes through the same cleaning
   and conversion steps, and the trained model predicts whether
   it's spam or not.

## Setup

### Prerequisites

- Python 3.x installed on your system
- Git (to clone the repo)

### Steps

1. Clone this repository

```bash
git clone https://github.com/ashutoshh-21/spam-sms-classifier
cd spam-sms-classifier
```

2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install the required packages

```bash
pip install -r requirements.txt
```

4. Download the dataset

   Go to [this Kaggle link](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset),
   download the CSV file, and place it inside the `data/` folder
   as `spam.csv`.

   Your folder should look like this:
   ```
   spam-sms-classifier/
   ├── data/
   │   └── spam.csv
   ├── models/
   ├── train.py
   ├── predict.py
   └── ...
   ```

5. Train the model (only needed once)

```bash
python train.py
```

   You should see output like:
   ```
   Loaded 5572 messages
   Ham: 4825, Spam: 747

   Cleaning text...
   Training model...

   Accuracy: 97.67%

   Detailed Results:
              precision    recall  f1-score   support

         Ham       0.97      1.00      0.99       965
        Spam       1.00      0.83      0.91       150

      accuracy                         0.98      1115
      macro avg       0.99      0.91   0.95      1115
   weighted avg       0.98      0.98   0.98      1115
   Model saved, run predict.py to test
   ```

6. Run the spam detector

```bash
python predict.py
```

   Type any message and press Enter. Type `quit` to exit.

## Project Structure

```
spam-sms-classifier/
├── data/
│   └── spam.csv            # the dataset
├── models/
│   ├── model.pkl           # trained model (created after running train.py)
│   └── vectorizer.pkl      # TF-IDF vectorizer (created after running train.py)
├── train.py                # script to train and save the model
├── predict.py              # CLI tool to classify messages
├── requirements.txt        # python dependencies
├── .gitignore
└── README.md
```

## Technologies Used

- **Python 3** — main programming language
- **pandas** — loading and handling the dataset
- **scikit-learn** — TF-IDF vectorizer and Naive Bayes classifier
- **NLTK** — text preprocessing (stopwords removal, stemming)

## Limitations

- Only works with English messages
- Trained on a relatively small dataset so it might not catch every
  type of spam
- Doesn't handle images or links inside messages, only the text content

## Future Improvements

- Try other models like SVM or Random Forest and compare results
- Add support for more languages
- Train on a larger and more recent dataset
- Handle slang and abbreviations better

## Acknowledgments

- Dataset: [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
  via Kaggle
- Built as a BYOP (Bring Your Own Project) for the Fundamentals of AI/ML course
