import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import nltk
import re
from nltk.corpus import stopwords

print("Starting model training setup...")
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# --- 1. SIMULATE DATA (In a real project, this would be your actual dataset) ---
data = [
    ("The battery life is excellent and the screen is clear.", "Positive"),
    ("The camera is awful and the customer support was rude.", "Negative"),
    ("It's an average product, nothing special.", "Neutral"),
    ("I love the durability and the fast delivery.", "Positive"),
    ("Too slow, overheats easily, waste of money.", "Negative"),
    ("Works fine for the price.", "Neutral"),
    ("Fantastic sound quality and ergonomic design.", "Positive"),
    ("The seller shipped the wrong color and ignored my request.", "Negative"),
]
df = pd.DataFrame(data, columns=['review', 'sentiment'])

# --- 2. PREPROCESSING FUNCTION ---
def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df['processed_review'] = df['review'].apply(preprocess)

# --- 3. TRAINING ---
X_train, X_test, y_train, y_test = train_test_split(df['processed_review'], df['sentiment'], test_size=0.3, random_state=42)

# Vectorizer (Feature Extraction)
vectorizer = TfidfVectorizer(max_features=100)
X_train_vec = vectorizer.fit_transform(X_train)

# Model Training
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# --- 4. SAVING THE MODEL AND VECTORIZER ---
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model and Vectorizer saved successfully!")
print("Run 'python app.py' to start the server.")