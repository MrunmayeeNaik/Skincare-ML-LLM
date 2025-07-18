import streamlit as st
import pandas as pd
import joblib
from chatbot import ask_gpt

# Load the dataset
df = pd.read_csv("skincare_remedies - Sheet1.csv")

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Title
st.title("🌿 Smart Natural Skincare Recommender")

# Ingredient selection
all_ingredients = sorted(set([ing.strip().lower() for i in df['Ingredients'] for ing in i.split(',')]))
selected_ingredients = st.multiselect("Select Ingredients You Have:", all_ingredients)

# Prediction function
def predict_skin_concern(user_ingredients):
    input_text = " ".join(user_ingredients)
    input_vector = vectorizer.transform([input_text])
    prediction = model.predict(input_vector)
    return prediction[0]

#LLM chatbot
st.header("💬 Ask our Skincare Chatbot")
chatbot_query = st.text_input("Ask me anything about your skin problems...",key="chatbot_query_input")

if st.button("Get Answer"):
    if chatbot_query:
        reply = ask_gpt(chatbot_query)
        st.success(reply)
#st.header("💬 Ask our Skincare Chatbot")
#user_query = st.text_input("Ask me anything about your skin problems...",key="user_query")

# Button
if st.button("Get Recommendation"):
    if selected_ingredients:
        concern = predict_skin_concern(selected_ingredients)
        st.success(f"Predicted Skin Concern: **{concern}**")

        # Filter matching remedies from dataset
        results = df[df['Skin Concern'].str.contains(concern, case=False, na=False)]
        if not results.empty:
            st.subheader("💡 Remedies You Can Try:")
            for _, row in results.iterrows():
                st.markdown(f"- **{row['Remedy Type']}**: {row['Remedy Description']}")
        else:
            st.warning("No exact remedy found for the predicted concern.")
    else:
        st.error("Please select at least one ingredient.")
