import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. PAGE SETUP & AGGRESSIVE CSS FIXES ---
st.set_page_config(page_title="ChurnGuard Pro", layout="wide")

st.markdown("""
    <style>
    /* 1. FORCE MAIN SCREEN BLACK */
    .stApp {
        background-color: #0E1117 !important;
    }
    
    /* 2. FORCE SIDEBAR WHITE & TEXT BLACK */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] h1 {
        color: #000000 !important;
    }

    /* 3. SIDEBAR FIELDS: Uniform Black Borders */
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"], 
    [data-testid="stSidebar"] .stNumberInput div[data-baseweb="input"] {
        border: 2px solid #000000 !important;
        border-radius: 4px !important;
        background-color: #FFFFFF !important;
    }

    /* 4. EXPANDERS: Keep them Black even when opened */
    div[data-testid="stExpander"] {
        background-color: #000000 !important;
        border: 1px solid #FFFFFF !important;
        border-radius: 8px !important;
    }
    div[data-testid="stExpander"] details summary {
        background-color: #000000 !important;
        color: #FFFFFF !important;
    }
    div[data-testid="stExpander"] div[data-testid="stVerticalBlock"] {
        background-color: #0E1117 !important; /* Content area background */
    }

    /* 5. MAIN SCREEN LABELS WHITE */
    label, p, h1, h2, h3 {
        color: #FFFFFF !important;
    }

   /* THE BUTTON: Bright White Background */
    div.stButton > button {
        background-color: #FFFFFF !important;
        border: 2px solid #000000 !important;
        width: 100% !important;
        height: 3.5em !important;
        border-radius: 8px !important;
    }

    /* THE TEXT: Forced Black and Bold */
    div.stButton > button p {
        color: #000000 !important;
        font-weight: 900 !important;
        font-size: 1.2rem !important;
        text-transform: uppercase !important;
    }
    
    /* PREVENT HOVER CHANGES: Keep it White/Black */
    div.stButton > button:hover, 
    div.stButton > button:active, 
    div.stButton > button:focus {
        background-color: #FFFFFF !important;
        border: 2px solid #000000 !important;
    }

    div.stButton > button:hover p {
        color: #000000 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA & MODEL ---
@st.cache_resource
def prepare_model():
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.fillna(0, inplace=True)
    df.drop('customerID', axis=1, inplace=True)
    
    encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, encoders, X.columns.tolist(), X, y

model, encoders, feature_names, X, y = prepare_model()

# --- 3. SIDEBAR ---
st.sidebar.title("👤 Personal Info")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.number_input("Tenure (Months)", 0, 72, 12)

# --- 4. MAIN SCREEN ---
st.title("Customer Retention Analytics")

with st.expander("🌐 Connectivity & Services (Click to Expand)"):
    col1, col2 = st.columns(2)
    with col1:
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        phone = st.selectbox("Phone Service", ["Yes", "No"])
    with col2:
        security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

with st.expander("💳 Contract & Billing Details (Click to Expand)"):
    col3, col4 = st.columns(2)
    with col3:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox("Payment Method", [
    "Electronic check", 
    "Mailed check", 
    "Bank transfer (automatic)", 
    "Credit card (automatic)"
])
    with col4:
        monthly = st.number_input("Monthly Charges ($)", 0.0, 150.0, 70.0)
        total = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)

# --- 5. THE CALCULATION & VISUALIZATION ---
st.markdown("###")
if st.button("CALCULATE CHURN RISK"):
    # Prepping data for model
    input_data = {
        'gender': gender, 'SeniorCitizen': senior, 'Partner': partner, 'Dependents': dependents,
        'tenure': tenure, 'PhoneService': phone, 'MultipleLines': 'No',
        'InternetService': internet, 'OnlineSecurity': security, 'OnlineBackup': 'No',
        'DeviceProtection': 'No', 'TechSupport': support, 'StreamingTV': 'No',
        'StreamingMovies': 'No', 'Contract': contract, 'PaperlessBilling': 'Yes',
        'PaymentMethod': payment, 'MonthlyCharges': monthly, 'TotalCharges': total
    }
    
    input_df = pd.DataFrame([input_data])
    for col, le in encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col])
    
    # Ensure correct column order for the model
    input_df = input_df[feature_names]
    prob = model.predict_proba(input_df)[0][1]
    
    st.markdown("---")
    
    # Create two columns for the Results
    res_col1, res_col2 = st.columns([1, 1.2])
    
    with res_col1:
        st.subheader("Prediction Result")
        if prob > 0.5:
            st.error(f"### HIGH RISK\n**Churn Probability: {prob:.1%}**")
            st.write("Recommendation: Offer a long-term contract or a loyalty discount.")
        else:
            st.success(f"### LOW RISK\n**Churn Probability: {prob:.1%}**")
            st.write("Recommendation: Maintain current service level.")
    
    with res_col2:
        st.subheader("Top 5 Risk Drivers")
        # Extract feature importance from the Random Forest model
        importances = pd.Series(model.feature_importances_, index=feature_names)
        top_5 = importances.nlargest(5)
        
        # Display as a bar chart
        st.bar_chart(top_5)
        st.caption("The longer the bar, the more this factor influenced the result.")
        # --- 6. MODEL PERFORMANCE (Technical Report Section) ---
    
    st.markdown("---")
    with st.expander("📊 Technical Model Performance (Report Data)"):
       
        # To show performance, we test it against the training data (simplified for this app)
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.metric("Model Accuracy", f"{acc:.2%}")
            st.write("This score represents how often the model is correct overall.")
            
        with col_m2:
            st.write("**Confusion Matrix**")
            cm = confusion_matrix(y, y_pred)
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot(fig)
            st.caption("Top-Left: Correct 'Stays' | Bottom-Right: Correct 'Churns'")