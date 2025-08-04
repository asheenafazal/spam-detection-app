import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Set page configuration
st.set_page_config(page_title="Spam Detector", page_icon="📩", layout="centered")

# Sidebar
st.sidebar.title("📊 Spam Detector Info")
st.sidebar.markdown("""
This app uses a **Naive Bayes** model trained on SMS spam data.

**Labels:**
- ✅ Ham (Not Spam)
- 🚫 Spam

Made with ❤️ using Streamlit
""")

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Train/test split
X = df['message']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model training
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Title and subtitle
st.markdown("<h1 style='text-align: center; color: #3366cc;'>📩 SMS Spam Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Type a message and see if it's spam or not!</p>", unsafe_allow_html=True)

# Input section
with st.form(key="spam_form"):
    message = st.text_area("✉️ Enter your SMS message:", height=150)
    submit = st.form_submit_button("🔍 Check Message")

if submit:
    if message.strip() == "":
        st.warning("⚠️ Please enter a message.")
    else:
        message_vector = vectorizer.transform([message])
        prediction = model.predict(message_vector)[0]

        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            if prediction == 1:
                st.error("🚫 **This message is SPAM!**")
            else:
                st.success("✅ **This message is NOT spam!**")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 12px;'>© 2025 Spam Detector | Built with 🧠 Streamlit</p>", unsafe_allow_html=True)
