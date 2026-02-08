import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("🧠 AI Mental Health Chat Analyzer")

user_input = st.text_area("How are you feeling today?")

if st.button("Analyze"):
    if user_input:
        transformed = vectorizer.transform([user_input])
        prediction = model.predict(transformed)[0]

        st.subheader(f"Detected Emotion: {prediction}")

        if prediction in ["depression", "stress"]:
            st.warning("⚠️ You might be experiencing emotional distress. Consider talking to someone you trust.")
        elif prediction == "happy":
            st.success("😊 Great to hear you're feeling positive!")
        else:
            st.info("Thank you for sharing your feelings.")
