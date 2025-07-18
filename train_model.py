import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load your dataset
df = pd.read_csv("skincare_remedies - Sheet1.csv")

# Define features (X) and target (y)
X = df['Ingredients'].astype(str)  # Input: ingredient list
y = df['Skin Concern']             # Output: skin concern

# Convert text to vectors
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train a simple model
model = MultinomialNB()
model.fit(X_vectorized, y)

# Save the trained model and vectorizer as binary files
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("✅ Model and vectorizer saved successfully.")
