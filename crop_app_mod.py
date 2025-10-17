import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --------------------
# USER AUTH SYSTEM
# --------------------
USER_DB = "users.csv"
HISTORY_DB = "recommendations.csv"

# Create databases if they don't exist
if not os.path.exists(USER_DB):
    pd.DataFrame(columns=["username", "password"]).to_csv(USER_DB, index=False)
if not os.path.exists(HISTORY_DB):
    pd.DataFrame(columns=["username", "N", "P", "K", "temperature", "humidity", "ph", "rainfall", "recommendation"]).to_csv(HISTORY_DB, index=False)

def load_users():
    return pd.read_csv(USER_DB)

def save_user(username, password):
    users = load_users()
    if username in users["username"].values:
        return False
    new_user = pd.DataFrame([[username, password]], columns=["username", "password"])
    users = pd.concat([users, new_user], ignore_index=True)
    users.to_csv(USER_DB, index=False)
    return True

def save_recommendation(username, input_data, recommendation):
    df = pd.read_csv(HISTORY_DB)
    entry = {
        "username": username,
        "N": input_data["N"].values[0],
        "P": input_data["P"].values[0],
        "K": input_data["K"].values[0],
        "temperature": input_data["temperature"].values[0],
        "humidity": input_data["humidity"].values[0],
        "ph": input_data["ph"].values[0],
        "rainfall": input_data["rainfall"].values[0],
        "recommendation": recommendation
    }
    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    df.to_csv(HISTORY_DB, index=False)

def load_user_history(username):
    df = pd.read_csv(HISTORY_DB)
    user_df = df[df["username"] == username]
    return user_df

# --------------------
# CROP INFO DATABASE
# --------------------
CROP_INFO = {
    "rice": "🌾 Rice thrives in hot and humid climates. It requires plenty of water and clayey or loamy soil. Ideal temperature: 20°C–37°C.",
    "maize": "🌽 Maize prefers warm climates and well-drained fertile soil with moderate rainfall. Ideal temperature: 18°C–27°C.",
    "chickpea": "🧆 Chickpeas grow best in cool, dry climates and well-drained loamy soil. Ideal for rabi (winter) season.",
    "kidneybeans": "🫘 Kidney beans prefer warm, frost-free climates. They grow best in well-drained loamy soil with moderate rainfall.",
    "pigeonpeas": "🌿 Pigeon peas are drought-resistant and grow well in tropical climates with moderate rainfall. Ideal temperature: 18°C–35°C.",
    "mothbeans": "🌱 Moth beans thrive in arid and semi-arid regions. They tolerate drought well and grow best in sandy or loamy soil.",
    "blackgram": "🌾 Black gram (urad dal) grows well in warm, humid climates with well-drained loamy soil. It’s often grown in the monsoon season.",
    "lentil": "🥣 Lentils prefer cool, dry climates and fertile, well-drained loamy soil. Ideal temperature: 10°C–25°C.",
    "pomegranate": "🍎 Pomegranates grow best in dry climates with moderate temperatures. They require well-drained sandy or loamy soil.",
    "grapes": "🍇 Grapes thrive in warm, dry climates with plenty of sunlight. They prefer well-drained, fertile soil. Ideal temperature: 15°C–35°C.",
    "watermelon": "🍉 Watermelons prefer hot, dry climates and sandy loam soil rich in organic matter. They need abundant sunlight.",
    "muskmelon": "🍈 Muskmelons grow well in warm climates with well-drained sandy soil. They need low humidity during fruit development.",
    "papaya": "🍍 Papayas grow well in tropical and subtropical climates with moderate rainfall. Ideal temperature: 25°C–35°C.",
    "coconut": "🥥 Coconuts thrive in coastal tropical regions with high humidity, abundant sunlight, and sandy, well-drained soil.",
    "jute": "🪶 Jute grows best in warm, humid climates with heavy rainfall and fertile alluvial soil. Ideal temperature: 24°C–35°C.",
    "orange": "🍊 Oranges grow in tropical and subtropical climates with moderate rainfall and plenty of sunlight. Ideal temperature: 15°C–30°C.",
    "cotton": "🧵 Cotton grows best in warm climates with low humidity. It prefers black soil or loamy soil and needs moderate rainfall.",
    "mungbean": "🌱 Mung beans (green gram) thrive in warm, moist climates with moderate rainfall and well-drained sandy loam soil.",
    "apple": "🍎 Apples grow in cool, temperate climates with cold winters. They need well-drained loamy soil and plenty of sunlight.",
    "banana": "🍌 Bananas require a warm, humid climate with high rainfall and rich loamy soil. They need consistent moisture throughout growth.",
    "mango": "🥭 Mango trees thrive in tropical and subtropical climates. They prefer dry weather during flowering and well-drained soil.",
    "coffee": "☕ Coffee grows in tropical regions with moderate sunlight, high humidity, and well-drained soil rich in organic matter."
}

# Initialize session
if "page" not in st.session_state:
    st.session_state.page = "login"

st.set_page_config(page_title="🌾 Crop Recommender Login", layout="centered")

# --------------------
# LOGIN PAGE
# --------------------
if st.session_state.page == "login":
    st.title("🔐 Login to Crop Recommendation System")

    username = st.text_input("👤 Username")
    password = st.text_input("🔑 Password", type="password")

    if st.button("Login"):
        users = load_users()
        if username in users["username"].values:
            if users.loc[users["username"] == username, "password"].values[0] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("✅ Login successful! Redirecting...")
                st.session_state.page = "app"
                st.rerun()
            else:
                st.error("❌ Incorrect password.")
        else:
            st.error("❌ User not found. Please sign up first.")

    st.markdown("Don't have an account?")
    if st.button("Sign Up"):
        st.session_state.page = "signup"
        st.rerun()

# --------------------
# SIGN-UP PAGE
# --------------------
elif st.session_state.page == "signup":
    st.title("📝 Create a New Account")

    new_user = st.text_input("👤 Choose a username")
    new_pass = st.text_input("🔑 Choose a password", type="password")

    if st.button("Create Account"):
        if new_user.strip() == "" or new_pass.strip() == "":
            st.warning("⚠️ Please enter both username and password.")
        else:
            if save_user(new_user, new_pass):
                st.success("🎉 Account created successfully! You can now log in.")
                st.session_state.page = "login"
                st.rerun()
            else:
                st.error("❌ Username already exists. Please choose another.")

    if st.button("Back to Login"):
        st.session_state.page = "login"
        st.rerun()

# --------------------
# ORIGINAL APP + POPUP INFO
# --------------------
elif st.session_state.page == "app":
    data = pd.read_csv("cropsrec.csv")

    X = data.drop('label', axis=1)
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.set_page_config(page_title="🌱 Crop Recommendation", layout="centered")

    st.title("🌾 Crop Recommendation System")
    st.markdown(f"Welcome, **{st.session_state.username}** 👋")
    st.markdown("Enter your soil and climate details to get the best crop suggestion.")

    N = st.number_input("Nitrogen (N)", min_value=0.0, step=1.0)
    P = st.number_input("Phosphorus (P)", min_value=0.0, step=1.0)
    K = st.number_input("Potassium (K)", min_value=0.0, step=1.0)
    temperature = st.number_input("Temperature (°C)", step=0.1)
    humidity = st.number_input("Humidity (%)", step=0.1)
    ph = st.number_input("Soil pH", step=0.1)
    rainfall = st.number_input("Rainfall (mm)", step=0.1)

    if st.button("Recommend Crop"):
        input_data = pd.DataFrame(
            [[N, P, K, temperature, humidity, ph, rainfall]],
            columns=X.columns
        )
        prediction = classifier.predict(input_data)
        crop = prediction[0]
        st.success(f"🌱 Recommended Crop: **{crop}**")
        save_recommendation(st.session_state.username, input_data, crop)

        # POPUP-STYLE MESSAGE
        info = CROP_INFO.get(crop.lower(), "No additional information available for this crop.")
        with st.expander(f"ℹ️ Learn more about {crop.title()}"):
            st.info(info)

    st.subheader("📊 Model Performance")
    st.write(f"✅ Accuracy: **{accuracy:.2f}**")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🕒 View My Recommendations"):
            st.session_state.page = "history"
            st.rerun()
    with col2:
        if st.button("🚪 Logout"):
            st.session_state.page = "login"
            st.session_state.logged_in = False
            st.rerun()

# --------------------
# USER HISTORY PAGE
# --------------------
elif st.session_state.page == "history":
    st.title("🕒 My Past Recommendations")
    st.markdown(f"User: **{st.session_state.username}**")

    history = load_user_history(st.session_state.username)

    if history.empty:
        st.info("No past recommendations found.")
    else:
        st.dataframe(history[["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "recommendation"]])

    if st.button("⬅️ Back to App"):
        st.session_state.page = "app"
        st.rerun()
