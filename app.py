import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Laptop Price Predictor 💻",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Premium CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    * {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Global Background Adjustments */
    .stApp {
        background-color: #0d0e15;
        color: #f7fafc;
    }
    
    /* Gradient Main Title */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 50%, #9B51E0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #a0aec0;
        margin-bottom: 2rem;
    }
    
    /* Premium Glassmorphism Cards */
    .premium-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 26px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    
    .premium-card:hover {
        transform: translateY(-5px);
        border-color: rgba(0, 242, 254, 0.4);
        box-shadow: 0 15px 35px 0 rgba(0, 242, 254, 0.15);
    }
    
    .card-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #00F2FE;
        font-weight: 600;
        margin-bottom: 8px;
    }
    
    .card-value {
        font-size: 2.8rem;
        font-weight: 800;
        color: #FFFFFF;
    }
    
    .card-desc {
        font-size: 0.85rem;
        color: #a0aec0;
        margin-top: 8px;
    }
    
    /* Metric Pills styling */
    .metric-pill-cyan {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        background: rgba(0, 242, 254, 0.08);
        color: #00F2FE;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 8px;
        margin-top: 6px;
        border: 1px solid rgba(0, 242, 254, 0.2);
    }
    
    .metric-pill-purple {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        background: rgba(155, 81, 224, 0.08);
        color: #BF8EFF;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 8px;
        margin-top: 6px;
        border: 1px solid rgba(155, 81, 224, 0.2);
    }
    
    /* Input UI Spacing adjustment */
    div[data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.01) !important;
        border-color: rgba(255, 255, 255, 0.05) !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="main-title">Laptop Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict laptop market prices based on hardware specifications using Linear Regression.</div>', unsafe_allow_html=True)

# --- Sidebar Configuration ---
st.sidebar.markdown("### ⚙️ Model Settings")

data_source = st.sidebar.radio(
    "Dataset Source:",
    ("Use Default Dataset", "Upload Custom CSV")
)

df = None
default_path = "laptop_prices.csv"

if data_source == "Use Default Dataset":
    if os.path.exists(default_path):
        df = pd.read_csv(default_path)
    else:
        st.sidebar.warning(f"Default dataset '{default_path}' not found. Please upload a CSV.")
else:
    uploaded_file = st.sidebar.file_uploader("Upload Laptop CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.sidebar.error(f"Error loading CSV file: {e}")

if df is not None:
    # Ensure columns exist
    required_cols = ["RAM", "SSD", "Processor_Gen", "Price"]
    columns_matched = all(col in df.columns for col in required_cols)
    
    if not columns_matched:
        st.sidebar.error(f"CSV must contain these exact columns: {', '.join(required_cols)}")
        st.sidebar.info(f"Loaded columns: {list(df.columns)}")
        st.stop()
        
    st.sidebar.markdown("### 🧪 Train-Test Split")
    test_size = st.sidebar.slider("Test Split Size (%)", min_value=10, max_value=50, value=20, step=5)
    random_seed = st.sidebar.number_input("Random Seed", min_value=0, max_value=1000, value=42)

    # --- Preprocessing & Linear Regression Training ---
    X = df[["RAM", "SSD", "Processor_Gen"]]
    y = df["Price"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100.0, random_state=random_seed
    )
    
    # Train the Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict and Calculate Performance Metrics
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    coef_ram = model.coef_[0]
    coef_ssd = model.coef_[1]
    coef_gen = model.coef_[2]
    intercept = model.intercept_

    # --- Layout: Main Navigation Tabs ---
    tab1, tab2, tab3 = st.tabs(["🚀 Price Predictor", "📈 Model Analytics", "📂 Dataset Explorer"])
    
    with tab1:
        col1, col2 = st.columns([1, 1.2], gap="large")
        
        with col1:
            st.markdown("### Choose Specifications")
            st.markdown("Adjust the controls below to configure your desired laptop hardware setup:")
            
            # Interactive sliders / input fields for hardware specs
            ram_val = st.slider("RAM Capacity (GB):", min_value=4, max_value=128, value=16, step=4)
            ssd_val = st.slider("SSD Storage (GB):", min_value=128, max_value=4096, value=512, step=128)
            gen_val = st.slider("Processor Generation (Gen):", min_value=8, max_value=15, value=12, step=1)
            
            # Prediction calculation
            input_features = pd.DataFrame({
                "RAM": [ram_val],
                "SSD": [ssd_val],
                "Processor_Gen": [gen_val]
            })
            
            predicted_price = model.predict(input_features)[0]
            # Ensure price prediction is non-negative
            predicted_price = max(predicted_price, 0.0)
            
            # Premium Prediction Display Card
            st.markdown(f"""
                <div class="premium-card">
                    <div class="card-label">Estimated Laptop Price</div>
                    <div class="card-value">${predicted_price:,.2f}</div>
                    <div class="card-desc">Computed for <b>{ram_val}GB RAM</b>, <b>{ssd_val}GB SSD</b>, and <b>Gen {gen_val}</b> Processor.</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Technical equation breakdown
            st.markdown("##### Math Behind the Prediction")
            st.latex(rf"\text{{Price}} = {coef_ram:.2f} \times \text{{RAM}} + {coef_ssd:.3f} \times \text{{SSD}} + {coef_gen:.2f} \times \text{{Processor\_Gen}} + {intercept:.2f}")
            st.markdown(f"""
                - **RAM Coefficient:** `{coef_ram:.3f}` *(Price increase per GB RAM)*
                - **SSD Coefficient:** `{coef_ssd:.3f}` *(Price increase per GB SSD)*
                - **Processor Gen Coefficient:** `{coef_gen:.3f}` *(Price increase per generation)*
                - **Intercept:** `{intercept:.3f}`
            """)
            
        with col2:
            st.markdown("### Specifications Breakdown")
            st.markdown("A visual representation of how your selected components contribute relative to the base intercept price.")
            
            # Horizontal breakdown bar chart
            intercept_contribution = intercept
            ram_contribution = coef_ram * ram_val
            ssd_contribution = coef_ssd * ssd_val
            gen_contribution = coef_gen * gen_val
            
            contrib_df = pd.DataFrame({
                "Component": ["Base Intercept", "RAM Contribution", "SSD Contribution", "Processor Gen Contribution"],
                "Value ($)": [intercept_contribution, ram_contribution, ssd_contribution, gen_contribution]
            })
            
            # Color palette matching theme
            fig_bar = px.bar(
                contrib_df, 
                x="Value ($)", 
                y="Component", 
                orientation="h",
                text="Value ($)",
                color="Component",
                color_discrete_sequence=["#2c3e50", "#00F2FE", "#4FACFE", "#9B51E0"]
            )
            fig_bar.update_traces(
                texttemplate='$%{text:,.2f}', 
                textposition='outside',
                hovertemplate="<b>%{y}</b><br>Contribution: $%{x:,.2f}<extra></extra>"
            )
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=10, b=0),
                showlegend=False,
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Price Influence ($)"),
                yaxis=dict(title=None)
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)
            
    with tab2:
        st.markdown("### Model Metrics & Performance")
        st.markdown("Evaluation metrics of the trained scikit-learn Linear Regression model on the test partition.")
        
        m_col1, m_col2, m_col3 = st.columns(3)
        
        with m_col1:
            st.markdown(f"""
                <div class="premium-card">
                    <div class="card-label">R-Squared (R²) Score</div>
                    <div class="card-value">{r2:.4f}</div>
                    <div class="card-desc">The goodness-of-fit. R² close to 1 indicates a very highly accurate model.</div>
                </div>
            """, unsafe_allow_html=True)
            
        with m_col2:
            st.markdown(f"""
                <div class="premium-card">
                    <div class="card-label">Mean Absolute Error (MAE)</div>
                    <div class="card-value">${mae:,.2f}</div>
                    <div class="card-desc">Average absolute error magnitude. Represents deviation from target pricing in dollars.</div>
                </div>
            """, unsafe_allow_html=True)
            
        with m_col3:
            st.markdown(f"""
                <div class="premium-card">
                    <div class="card-label">Root Mean Squared Error (RMSE)</div>
                    <div class="card-value">${rmse:,.2f}</div>
                    <div class="card-desc">Standard deviation of the residuals. Larger errors are more heavily penalized.</div>
                </div>
            """, unsafe_allow_html=True)
            
        # Model splits info
        st.markdown("##### Split Distribution Details")
        st.markdown(f"""
        <span class="metric-pill-cyan">Training Samples: {len(X_train)}</span>
        <span class="metric-pill-purple">Testing Samples: {len(X_test)}</span>
        <span class="metric-pill-cyan">Test Split Size: {test_size}%</span>
        <span class="metric-pill-purple">Random Seed: {random_seed}</span>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Performance Charts
        c_col1, c_col2 = st.columns(2, gap="large")
        
        with c_col1:
            st.markdown("##### Actual Price vs Predicted Price")
            st.markdown("Points close to the diagonal line represent accurate predictions.")
            
            fit_df = pd.DataFrame({
                "Actual Price": y_test.values,
                "Predicted Price": y_pred
            })
            
            fig_scatter = px.scatter(
                fit_df,
                x="Actual Price",
                y="Predicted Price",
                labels={"Actual Price": "Actual Price ($)", "Predicted Price": "Predicted Price ($)"},
                trendline="ols",
                trendline_color_override="#00F2FE"
            )
            fig_scatter.update_traces(
                marker=dict(size=8, color="#9B51E0", opacity=0.7, line=dict(width=1, color="white")),
                hovertemplate="Actual: $%{x:,.2f}<br>Predicted: $%{y:,.2f}<extra></extra>"
            )
            fig_scatter.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        with c_col2:
            st.markdown("##### Feature Importance (Coefficients)")
            st.markdown("Represents the pricing increase per unit change in hardware specifications.")
            
            feat_df = pd.DataFrame({
                "Hardware Feature": ["RAM (per GB)", "SSD (per GB)", "Processor (per Gen)"],
                "Coefficient ($)": [coef_ram, coef_ssd, coef_gen]
            })
            
            fig_feat = px.bar(
                feat_df,
                x="Hardware Feature",
                y="Coefficient ($)",
                text="Coefficient ($)",
                color="Hardware Feature",
                color_discrete_sequence=["#00F2FE", "#4FACFE", "#9B51E0"]
            )
            fig_feat.update_traces(
                texttemplate='$%{text:,.2f}',
                textposition='outside',
                hovertemplate="<b>%{x}</b><br>Coeff: $%{y:,.2f}<extra></extra>"
            )
            fig_feat.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=10, b=0),
                showlegend=False,
                xaxis=dict(title=None),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Coeff Weight ($)")
            )
            st.plotly_chart(fig_feat, use_container_width=True)

    with tab3:
        st.markdown("### Raw Dataset Explorer")
        st.markdown("Inspect, filter, or download the active dataset used to train the Laptop Price Predictor.")
        
        d_col1, d_col2 = st.columns([1, 2], gap="large")
        
        with d_col1:
            st.markdown("##### Summary Statistics")
            st.dataframe(df.describe())
            
        with d_col2:
            st.markdown("##### Raw Data View")
            st.dataframe(df, height=350, use_container_width=True)
            
            # CSV Download Button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Laptop Price Dataset (CSV)",
                data=csv,
                file_name='laptop_prices.csv',
                mime='text/csv',
            )

else:
    st.info("👋 Welcome! Please upload a laptop CSV dataset using the sidebar to train the linear model, or select the default option.")
