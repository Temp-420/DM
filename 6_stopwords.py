import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Sample data (text, label)
data = [
    ("Free tickets to the concert!!!", "spam"),
    ("Call now and win a prize", "spam"),
    ("Hey, are we meeting tomorrow?", "ham"),
    ("Can you send me the notes?", "ham"),
    ("Congratulations! You have won", "spam"),
    ("Let’s catch up this weekend", "ham")
]

# Separate features and labels
texts, labels = zip(*data)

# Preprocessing function
def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove numbers/symbols
    text = text.lower()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Preprocess all messages
cleaned_texts = [preprocess(t) for t in texts]

# Bag of Words
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(cleaned_texts)
y = labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict & evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
