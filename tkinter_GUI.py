import tkinter as tk
from tkinter import messagebox
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

# Load and preprocess the dataset
df = pd.read_csv('D:\Internship tasks\Task1\IMDB Dataset.csv')
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

# Function to predict sentiment
def predict_sentiment():
    user_review = review_input.get("1.0", "end-1c")
    if user_review:
        processed_review = preprocess_review(user_review)
        review_vector = vectorizer.transform([processed_review])
        prediction = model.predict(review_vector)
        sentiment = "Positive" if prediction[0] == 'positive' else "Negative"
        messagebox.showinfo("Prediction", f"Sentiment: {sentiment}")
    else:
        messagebox.showwarning("Input Error", "Please enter a review to predict sentiment.")

# Set up Tkinter window
window = tk.Tk()
window.title("Movie Review Sentiment Analysis")
window.geometry("400x300")

# Add a label
label = tk.Label(window, text="Enter your movie review below:")
label.pack(pady=10)

# Add a Text widget to take user input (movie review)
review_input = tk.Text(window, height=5, width=40)
review_input.pack(pady=10)

# Add a Button to predict sentiment
predict_button = tk.Button(window, text="Predict Sentiment", command=predict_sentiment)
predict_button.pack(pady=10)

# Add a Label to display model accuracy and F1 score
accuracy_label = tk.Label(window, text=f"Accuracy: {accuracy:.2f}")
accuracy_label.pack(pady=5)

f1_label = tk.Label(window, text=f"F1 Score: {f1:.2f}")
f1_label.pack(pady=5)

# Run the Tkinter event loop
window.mainloop()
