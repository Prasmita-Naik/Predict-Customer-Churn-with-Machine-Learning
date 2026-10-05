import streamlit as st
import pandas as pd
import pickle

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Churn PredictorS",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #07111f 0%, #0d1b2a 45%, #111827 100%);
        color: white;
    }

    /* Hide default Streamlit menu */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    /* Main title */
    .main-title {
        font-size: 52px;
        font-weight: 800;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 5px;
        background: linear-gradient(90deg, #00ff9d, #7c3aed, #00d9ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        text-align: center;
        font-size: 19px;
        color: #b8c4d6;
        margin-bottom: 35px;
    }

    /* Cards */
    .card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.20);
    }

    .card-title {
        font-size: 23px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 8px;
    }

    .card-text {
        color: #b8c4d6;
        font-size: 15px;
    }

    /* Prediction result */
    .prediction-card {
        padding: 35px;
        border-radius: 22px;
        text-align: center;
        margin-top: 25px;
        margin-bottom: 25px;
    }

    .churn-result {
        background: linear-gradient(
            135deg,
            rgba(239,68,68,0.20),
            rgba(127,29,29,0.35)
        );
        border: 1px solid rgba(239,68,68,0.5);
    }

    .safe-result {
        background: linear-gradient(
            135deg,
            rgba(0,255,157,0.12),
            rgba(5,150,105,0.25)
        );
        border: 1px solid rgba(0,255,157,0.45);
    }

    .prediction-icon {
        font-size: 60px;
    }

    .prediction-title {
        font-size: 35px;
        font-weight: 800;
        margin-top: 10px;
    }

    .prediction-text {
        font-size: 18px;
        color: #d1d5db;
        margin-top: 8px;
    }

    /* Metric cards */
    .metric-card {
        background: rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
    }

    .metric-number {
        font-size: 30px;
        font-weight: 800;
        color: #00ff9d;
    }

    .metric-label {
        font-size: 14px;
        color: #9ca3af;
    }

    /* Section headers */
    .section-header {
        font-size: 28px;
        font-weight: 750;
        color: white;
        margin-top: 15px;
        margin-bottom: 18px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #08111f, #0c1727);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        border: none;
        padding: 12px 20px;
        font-weight: 700;
        background: linear-gradient(90deg, #00c896, #7c3aed);
        color: white;
        transition: 0.3s;
    }

    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 25px rgba(124,58,237,0.35);
    }

    /* Input boxes */
    .stNumberInput input,
    .stSelectbox div {
        border-radius: 10px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #6b7280;
        padding: 30px;
        font-size: 13px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    with open("rf_model.pkl", "rb") as file:
        model_data = pickle.load(file)

    # Your notebook saved model + features together
    if isinstance(model_data, dict):

        model = model_data["model"]
        features = model_data["features"]

    else:

        # Backup in case only the model was saved
        model = model_data

        features = [
            "Call  Failure",
            "Complains",
            "Subscription  Length",
            "Charge  Amount",
            "Seconds of Use",
            "Frequency of use",
            "Frequency of SMS",
            "Distinct Called Numbers",
            "Age Group",
            "Tariff Plan",
            "Status",
            "Customer Value"
        ]

except FileNotFoundError:

    st.error(
        "⚠️ rf_model.pkl was not found. "
        "Please keep rf_model.pkl in the same folder as app.py."
    )

    st.stop()

except Exception as e:

    st.error(f"Unable to load the model: {e}")
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 📡 ChurnGuard AI")

    st.markdown(
        """
        <div style="
            background: rgba(255,255,255,0.05);
            padding: 18px;
            border-radius: 15px;
            margin-top: 10px;
        ">
        <b>AI Customer Churn Predictor</b><br><br>
        Predict whether a customer is likely to leave the service
        using a Machine Learning model.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### 🤖 Model")

    st.info("Random Forest Classifier")

    st.markdown("### 📊 Model Inputs")

    st.write("12 customer features")

    st.markdown("---")

    st.markdown("### 🧠 How it works")

    st.write("1️⃣ Enter customer information")
    st.write("2️⃣ AI analyzes the information")
    st.write("3️⃣ Model predicts churn")
    st.write("4️⃣ View prediction confidence")

    st.markdown("---")

    st.caption("Customer Churn Prediction Project")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📡 ChurnGuard AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict Customer Churn with Machine Learning'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# TOP METRICS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-number">12</div>
            <div class="metric-label">Customer Features</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-number">RF</div>
            <div class="metric-label">Machine Learning Model</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-number">AI</div>
            <div class="metric-label">Churn Prediction</div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# CUSTOMER INFORMATION
# ============================================================

st.markdown(
    '<div class="section-header">👤 Customer Information</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="card-text">'
    'Enter the customer information below. '
    'The AI model will analyze these characteristics.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# INPUT FORM
# ============================================================

with st.form("churn_prediction_form"):

    # --------------------------------------------------------
    # SECTION 1
    # --------------------------------------------------------

    st.markdown("### 📞 Usage & Calling")

    col1, col2, col3 = st.columns(3)

    with col1:

        call_failure = st.number_input(
            "Call Failure",
            min_value=0.0,
            value=0.0,
            step=1.0,
            help="Number of failed calls"
        )

    with col2:

        subscription_length = st.number_input(
            "Subscription Length",
            min_value=0.0,
            value=12.0,
            step=1.0,
            help="Length of the customer's subscription"
        )

    with col3:

        seconds_use = st.number_input(
            "Seconds of Use",
            min_value=0.0,
            value=1000.0,
            step=10.0,
            help="Total seconds of service usage"
        )


    # --------------------------------------------------------
    # SECTION 2
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("### 💬 Customer Activity")

    col1, col2, col3 = st.columns(3)

    with col1:

        complains = st.selectbox(
            "Complains",
            options=[0, 1],
            format_func=lambda x:
                "No Complaints" if x == 0 else "Has Complaints"
        )

    with col2:

        frequency_use = st.number_input(
            "Frequency of Use",
            min_value=0.0,
            value=50.0,
            step=1.0
        )

    with col3:

        frequency_sms = st.number_input(
            "Frequency of SMS",
            min_value=0.0,
            value=10.0,
            step=1.0
        )


    # --------------------------------------------------------
    # SECTION 3
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("### 📱 Customer Profile")

    col1, col2, col3 = st.columns(3)

    with col1:

        distinct_numbers = st.number_input(
            "Distinct Called Numbers",
            min_value=0.0,
            value=10.0,
            step=1.0
        )

    with col2:

        age_group = st.selectbox(
            "Age Group",
            options=[1, 2, 3, 4, 5],
            help="Age group category used in the original dataset"
        )

    with col3:

        customer_value = st.number_input(
            "Customer Value",
            min_value=0.0,
            value=100.0,
            step=1.0
        )


    # --------------------------------------------------------
    # SECTION 4
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("### 💳 Plan & Account Details")

    col1, col2, col3 = st.columns(3)

    with col1:

        charge_amount = st.number_input(
            "Charge Amount",
            min_value=0.0,
            value=5.0,
            step=1.0
        )

    with col2:

        tariff_plan = st.selectbox(
            "Tariff Plan",
            options=[1, 2]
        )

    with col3:

        status = st.selectbox(
            "Customer Status",
            options=[1, 2]
        )


    # --------------------------------------------------------
    # SUBMIT
    # --------------------------------------------------------

    st.markdown("<br>", unsafe_allow_html=True)

    submitted = st.form_submit_button(
        "🚀 PREDICT CUSTOMER CHURN"
    )


# ============================================================
# PREDICTION
# ============================================================

if submitted:

    # Create input DataFrame
    input_data = pd.DataFrame(
        [[
            call_failure,
            complains,
            subscription_length,
            charge_amount,
            seconds_use,
            frequency_use,
            frequency_sms,
            distinct_numbers,
            age_group,
            tariff_plan,
            status,
            customer_value
        ]],
        columns=features
    )


    # Make prediction
    prediction = model.predict(input_data)[0]


    # Get probability if supported
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(input_data)[0]

        no_churn_probability = probabilities[0] * 100
        churn_probability = probabilities[1] * 100

    else:

        no_churn_probability = 100 if prediction == 0 else 0
        churn_probability = 100 if prediction == 1 else 0


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-header">🔮 Prediction Result</div>',
        unsafe_allow_html=True
    )


    if prediction == 1:

        st.markdown(
            f"""
            <div class="prediction-card churn-result">

                <div class="prediction-icon">⚠️</div>

                <div class="prediction-title">
                    HIGH CHURN RISK
                </div>

                <div class="prediction-text">
                    The model predicts that this customer is
                    likely to churn.
                </div>

                <br>

                <div style="font-size:22px;font-weight:700;">
                    Churn Probability: {churn_probability:.1f}%
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.warning(
            "💡 This customer may require retention attention."
        )

    else:

        st.markdown(
            f"""
            <div class="prediction-card safe-result">

                <div class="prediction-icon">✅</div>

                <div class="prediction-title">
                    LOW CHURN RISK
                </div>

                <div class="prediction-text">
                    The model predicts that this customer
                    is likely to remain with the service.
                </div>

                <br>

                <div style="font-size:22px;font-weight:700;">
                    Retention Probability: {no_churn_probability:.1f}%
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.success(
            "🎉 This customer appears to have a low risk of churn."
        )


    # ========================================================
    # PROBABILITY BREAKDOWN
    # ========================================================

    st.markdown("### 📊 Prediction Confidence")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "No Churn Probability",
            f"{no_churn_probability:.1f}%"
        )

        st.progress(
            int(no_churn_probability)
        )

    with col2:

        st.metric(
            "Churn Probability",
            f"{churn_probability:.1f}%"
        )

        st.progress(
            int(churn_probability)
        )


    # ========================================================
    # INPUT SUMMARY
    # ========================================================

    st.markdown("---")

    st.markdown("### 📋 Customer Data Used")

    display_data = input_data.T

    display_data.columns = ["Value"]

    st.dataframe(
        display_data,
        use_container_width=True
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("🧠 About this AI Model"):

    st.write(
        """
        This application uses a Random Forest Classifier trained
        on customer usage and profile information.

        The model analyzes 12 customer features and predicts
        whether the customer is likely to churn.

        Prediction:

        • 0 → No Churn
        • 1 → Churn
        """
    )

    st.write("### Features expected by the model")

    for feature in features:

        st.write(f"• {feature}")


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        📡 ChurnGuard AI &nbsp; | &nbsp;
        Customer Churn Prediction System
        <br>
        Built with Python • Scikit-learn • Pandas • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)