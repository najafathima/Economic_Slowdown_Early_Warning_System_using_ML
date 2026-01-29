import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Economic Slowdown Early Warning System",
    layout="centered"
)

# --------------------------------------------------
# Dark theme styling
# --------------------------------------------------
st.markdown(
    """
    <style>
    body { background-color: #0e1117; color: white; }
    .stApp { background-color: #0e1117; }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# Title
# --------------------------------------------------
st.markdown(
    """
    <h1 style="text-align: center; margin-bottom: 0;">
        Economic Slowdown Early Warning System
    </h1>
    <p style="text-align: center; font-size: 18px; color: #cfcfcf; margin-top: 5px;">
        An intelligent decision-support system for early detection of
        economic slowdown signals.
    </p>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Load model & dataset
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("final_model.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("macro_data.csv")

    # Normalize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Safety check
    required_cols = ["year", "country_code"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    return df


model = load_model()
df = load_data()



# --------------------------------------------------
# Risk band function
# --------------------------------------------------
def slowdown_risk_band(prob):
    if prob < 0.30:
        return "🟢 Low risk (stable outlook)"
    elif prob < 0.50:
        return "🟡 Moderate risk"
    elif prob < 0.70:
        return "🟠 High risk"
    else:
        return "🔴 Very high risk"

# --------------------------------------------------
# User Inputs (ONLY 5 FEATURES)
# --------------------------------------------------
st.subheader("Enter the Inputs")

year = st.selectbox("Year", sorted(df["year"].unique()))
country = st.selectbox("Country", sorted(df["country_code"].unique()))

gdp_growth = st.number_input("GDP Growth ", -15.0, 15.0, 5.0)
inflation = st.number_input("Inflation ", 0.0, 25.0, 5.0)
exports_gdp = st.number_input("Exports (% of GDP)", -30.0, 60.0, 15.0)
population_growth = st.number_input("Population Growth ", -3.0, 5.0, 1.2)
current_account = st.number_input("Current Account Balance (% of GDP)", -20.0, 20.0, -2.0
)

# --------------------------------------------------
# Predict
# --------------------------------------------------
if st.button("🔍 Predict Economic Slowdown Risk"):

    # ---- Fetch base row from dataset
    base_row = df[
        (df["year"] == year) &
        (df["country_code"] == country)
    ]

    if base_row.empty:
        st.error("No data available for the selected year and country.")
        st.stop()

    # ---- Drop target column
    X_input = base_row.drop(columns=["slowdown_score"], errors="ignore").copy()

    # ---- Override ONLY selected lag1 features
    X_input.loc[:, "gdp_growth_lag1"] = gdp_growth
    X_input.loc[:, "inflation_lag1"] = inflation
    X_input.loc[:, "exports_gdp_lag1"] = exports_gdp
    X_input.loc[:, "population_growth_lag1"] = population_growth
    X_input.loc[:, "current_account_balance_lag1"] = current_account

    # --------------------------------------------------
    # Model prediction (core ML step)
    # --------------------------------------------------
    prob = model.predict_proba(X_input)[0, 1]
    prob_pct = round(prob * 100, 2)

    # --------------------------------------------------
    # Output
    # --------------------------------------------------
    st.subheader("📊 Prediction Result")

    st.metric(
        label="Probability of Economic Slowdown",
        value=f"{prob_pct} %"
    )

    st.markdown(
        f"### Risk Assessment: **{slowdown_risk_band(prob)}**"
    )
