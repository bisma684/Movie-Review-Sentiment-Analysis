import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK datasets
nltk.download('stopwords')
nltk.download('punkt')

# Preprocessing function
stop_words = set(stopwords.words('english'))
def preprocess_review(review):
    words = word_tokenize(review)
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(words)

# Load and preprocess the dataset (use your path to the Kaggle dataset)
df = pd.read_csv('IMDB Dataset.csv')
df['processed_review'] = df['review'].apply(preprocess_review)

# Convert text data into numerical features using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['processed_review'])
y = df['sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, pos_label='positive')

# Streamlit user interface
st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review below, and the model will predict whether the sentiment is positive or negative.")

# Text input field for the review
user_review = st.text_area("Enter your movie review here:")

# Prediction button
if st.button('Predict Sentiment'):
    if user_review:
        processed_review = preprocess_review(user_review)
        review_vector = vectorizer.transform([processed_review])
        prediction = model.predict(review_vector)
        sentiment = "Positive" if prediction[0] == 'positive' else "Negative"
        st.write(f"Sentiment: {sentiment}")
    else:
        st.write("Please enter a review to predict sentiment.")

# Display accuracy and F1 score of the model
st.sidebar.subheader("Model Evaluation:")
st.sidebar.write(f"Accuracy: {accuracy}")
st.sidebar.write(f"F1 Score: {f1}")
