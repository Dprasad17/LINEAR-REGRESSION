import streamlit as st
import pandas as pd
import joblib

# ==============================================================================
# PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏡",
    layout="centered",
)

st.markdown(
    """
    <style>
    .main-header {
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 0.3rem;
    }
    .sub-title {
        color: #666;
        font-size: 1rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1px solid #86efac;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .price-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #15803d;
        margin: 8px 0;
    }
    .price-range {
        font-size: 1.15rem;
        font-weight: 600;
        color: #166534;
    }
    .range-caption {
        font-size: 0.85rem;
        color: #4b5563;
        margin-top: 4px;
    }
    .metrics-header {
        margin-top: 2rem;
        margin-bottom: 0.75rem;
        font-weight: 700;
        font-size: 1.15rem;
        color: inherit;
    }
    /* Prevent metric values from truncating with ellipsis */
    [data-testid="stMetric"] {
        overflow: visible !important;
        min-width: 0 !important;
    }
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] > div {
        font-size: 1.2rem !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: unset !important;
        letter-spacing: -0.02em;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.92rem !important;
        font-weight: 600 !important;
    }
    @media (max-width: 640px) {
        [data-testid="stMetricValue"], [data-testid="stMetricValue"] > div {
            font-size: 1.05rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# LOAD PRE-TRAINED MODEL FROM NOTEBOOK (CACHED)
# ==============================================================================
@st.cache_resource
def load_trained_model():
    """Loads the trained Linear Regression model and parameters saved from the notebook."""
    package = joblib.load("house_model.pkl")
    return (
        package["model"],
        package["lower_bound"],
        package["upper_bound"],
        package["metrics"],
    )

model, lower_bound, upper_bound, metrics = load_trained_model()

# ==============================================================================
# HEADER
# ==============================================================================
st.markdown(
    """
    <div class="main-header">
        <div class="main-title">🏡 House Price Predictor</div>
        <div class="sub-title">Enter house details below to estimate the price and view model performance metrics.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# INPUT FORM
# ==============================================================================
with st.container(border=True):
    st.subheader("Property Details")
    
    col1, col2 = st.columns(2)

    with col1:
        gr_liv_area = st.number_input(
            "Living Area (sq ft)",
            min_value=300,
            max_value=6000,
            value=1500,
            step=25,
            help="Above grade living area in square feet.",
        )
        overall_qual = st.slider(
            "Overall Quality (1 - 10)",
            min_value=1,
            max_value=10,
            value=6,
            help="Rates overall material and finish quality.",
        )
        total_bsmt_sf = st.number_input(
            "Basement Area (sq ft)",
            min_value=0,
            max_value=4000,
            value=1000,
            step=25,
            help="Total basement area in square feet.",
        )

    with col2:
        garage_area = st.number_input(
            "Garage Area (sq ft)",
            min_value=0,
            max_value=1500,
            value=480,
            step=10,
            help="Size of garage in square feet.",
        )
        full_bath = st.selectbox(
            "Full Bathrooms",
            options=[1, 2, 3, 4],
            index=1,
            help="Number of full bathrooms above ground.",
        )
        year_built = st.number_input(
            "Year Built",
            min_value=1870,
            max_value=2010,
            value=1995,
            step=1,
            help="Original construction year.",
        )

    predict_btn = st.button("Predict Price", type="primary", width="stretch")

# ==============================================================================
# PREDICTION & PRICE RANGE
# ==============================================================================
if predict_btn:
    # Prepare input dataframe
    user_inputs = pd.DataFrame([{
        "Gr Liv Area": float(gr_liv_area),
        "Overall Qual": float(overall_qual),
        "Total Bsmt SF": float(total_bsmt_sf),
        "Garage Area": float(garage_area),
        "Full Bath": float(full_bath),
        "Year Built": float(year_built),
    }])

    # Apply training IQR clipping from notebook
    clipped_inputs = user_inputs.clip(lower=lower_bound, upper=upper_bound, axis=1)

    # Compute prediction
    predicted_price = model.predict(clipped_inputs)[0]

    # Calculate expected price range using Test MAE
    mae = metrics["test_mae"]
    lower_range = max(0, predicted_price - mae)
    upper_range = predicted_price + mae

    # Render result card
    st.markdown(
        f"""
        <div class="prediction-box">
            <div style="font-size: 0.95rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #166534;">
                Estimated Sale Price
            </div>
            <div class="price-value">{predicted_price:,.2f}</div>
            <div class="price-range">
                Expected Price Range: {lower_range:,.0f} &ndash; {upper_range:,.0f}
            </div>
            <div class="range-caption">
                Based on model Mean Absolute Error (&plusmn;{mae:,.2f})
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==============================================================================
# MODEL PERFORMANCE METRICS
# ==============================================================================
st.markdown('<div class="metrics-header">📊 Model Evaluation Metrics</div>', unsafe_allow_html=True)

with st.container(border=True):
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    col_m1.metric("R² Score", f"{metrics['test_r2']:.4f}")
    col_m2.metric("MAE", f"{metrics['test_mae']:,.2f}")
    col_m3.metric("RMSE", f"{metrics['test_rmse']:,.2f}")
    col_m4.metric("MSE", f"{metrics['test_mse']:,.0f}")
