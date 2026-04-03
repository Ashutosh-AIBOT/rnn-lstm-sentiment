import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dashboard_core import (
    predict_sentiment, get_training_metrics, 
    get_pca_data, get_engineering_report, get_project_story
)

# --- 1. APP CONFIG ---
st.set_page_config(
    page_title="Sentiment LSTM Analysis (Custom Architecture)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. PREMIUM CSS (PROFESSIONAL CLINICAL THEME) ---
st.markdown("""
<style>
    #MainMenu {visibility: hidden}
    footer {visibility: hidden}
    header {visibility: hidden}
    
    [data-testid="stSidebar"] {background-color: #fcfcfc; border-right: 1px solid #eee}
    [data-testid="stSidebar"] * {color: #333; font-family: 'DM Sans', sans-serif}
    
    .block-container {padding-top: 2rem; padding-bottom: 2rem}
    h1, h2, h3 {font-weight: 500; letter-spacing: -0.02em; color: #1a1a1a}
    .muted-text {color: #666; font-size: 0.85rem}
    
    /* Professional KPI Cards */
    .kpi-row {display: flex; gap: 1.5rem; margin-bottom: 2.5rem}
    .kpi-card {
        flex: 1;
        padding: 1.2rem;
        background: #ffffff;
        border-radius: 4px;
        border: 1px solid #eee;
        border-bottom: 3px solid #7F77DD;
    }
    .kpi-label {font-size: 11px; color: #666; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em}
    .kpi-value {font-size: 24px; font-weight: 500; color: #1a1a1a; margin-top: 4px}

    /* Result Pill */
    .result-pill {
        padding: 10px 16px;
        background: #f9f9f9;
        border: 1px solid #eee;
        border-radius: 4px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .status-dot {width: 10px; height: 10px; border-radius: 50%}
    
    /* Strategy Section */
    .strategy-card {
        padding: 1rem;
        background: #fcfcfc;
        border: 1px solid #eee;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. SIDEBAR (ARCHITECTURE SPECS) ---
with st.sidebar:
    st.markdown("### LSTM CLASSIFIER v2.0")
    st.markdown("<p class='muted-text'>Custom Neural Architecture (Trained from Scratch)</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Model Architecture
    st.markdown("#### Hardware & Stack")
    st.markdown("""
    - **Architecture**: Stacked Bi-LSTM
    - **Training Mode**: From Scratch (No Transfer Learning)
    - **Embedding**: 128-dim Learned Vector
    - **Hidden State**: 256-dim Short-term Memory
    - **Optimization**: Gradient Clipping (1.0)
    - **Dataset**: IMDB Movie Reviews
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Clinical Parameters")
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
    st.caption("Lowering threshold increases sensitivity to weak sentiment signals.")

# --- 4. HEADER KPIs ---
eng_report = get_engineering_report()
st.markdown("## Sentiment Analysis Platform")
st.markdown("<p class='muted-text'>Neural-linguistic sentiment scoring for large-scale review datasets.</p>", unsafe_allow_html=True)

st.markdown(f"""
<div class="kpi-row">
    <div class="kpi-card">
        <div class="kpi-label">Performance Delta</div>
        <div class="kpi-value" style="color: #1D9E75">{eng_report['delta']}</div>
        <div class="muted-text">Above Random Baseline</div>
    </div>
    <div class="kpi-card" style="border-bottom-color: #7F77DD">
        <div class="kpi-label">Total Parameters</div>
        <div class="kpi-value">{eng_report['params']}</div>
        <div class="muted-text">Managed Embedding Matrix</div>
    </div>
    <div class="kpi-card" style="border-bottom-color: #333">
        <div class="kpi-label">Convergence Target</div>
        <div class="kpi-value">{eng_report['target_acc']}</div>
        <div class="muted-text">{eng_report['training_mode']}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 5. FUNCTIONAL TABS ---
tab1, tab2, tab3 = st.tabs(["Sentiment Inference", "Model Dynamics", "Engineering & Audit"])

with tab1:
    col_inp, col_out = st.columns([1.2, 1], gap="large")
    
    with col_inp:
        review_text = st.text_area("Write or paste a movie review...", height=200, placeholder="Example: This movie was a masterpiece. The acting was superb and the direction was flawless.")
        analyze_btn = st.button("Analyze Sentiment", type="primary", use_container_width=True)

    with col_out:
        if analyze_btn and review_text:
            with st.spinner("Executing linguistic forward pass..."):
                label, prob = predict_sentiment(review_text)
            
            # Badge
            status_color = "#1D9E75" if label == "Positive" else "#E24B4A"
            st.markdown(f"""
            <div class="result-pill">
                <div class="status-dot" style="background: {status_color}"></div>
                <div style="flex-grow: 1; font-weight: 500">{label.upper()}</div>
                <div style="color: #666; font-size: 13px">{prob*100:.1f}% Confidence</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability Gauge
            st.markdown("<br>", unsafe_allow_html=True)
            p_val = int(prob * 100)
            st.markdown(f"""
            <div class="kpi-label">Sentiment Probability Meter</div>
            <div style="height: 10px; width: 100%; background: #eee; border-radius: 5px; overflow: hidden; margin-top: 8px">
                <div style="height: 100%; width: {p_val}%; background: {status_color}"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 10px; color: #999; margin-top: 4px">
                <span>Negative</span>
                <span>Positive</span>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("#### Morphological Insight")
            st.caption("The model classifies sentiment by analyzing the sequential context of tokens. Positive sentiment often clusters toward the end of reviews in the IMDB dataset.")
        else:
            st.info("Awaiting review sequence to begin forward pass.")

with tab2:
    st.markdown("#### Learning & Optimization Evolution")
    history = get_training_metrics()
    if history:
        c1, c2 = st.columns(2)
        
        # Loss
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(y=history["train_loss"], name="Train", line=dict(color="#7F77DD", width=2)))
        fig_loss.add_trace(go.Scatter(y=history["val_loss"], name="Validation", line=dict(color="#333", width=1, dash="dot")))
        fig_loss.update_layout(title="Log Loss Curve", template="plotly_white", margin=dict(l=0,r=0,t=40,b=0), height=350)
        c1.plotly_chart(fig_loss, use_container_width=True)
        
        # Accuracy
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(y=history["train_acc"], name="Train", line=dict(color="#1D9E75", width=2)))
        fig_acc.add_trace(go.Scatter(y=history["val_acc"], name="Validation", line=dict(color="#333", width=1, dash="dot")))
        fig_acc.update_layout(title="Categorical Accuracy", template="plotly_white", margin=dict(l=0,r=0,t=40,b=0), height=350)
        c2.plotly_chart(fig_acc, use_container_width=True)
    else:
        st.warning("No metrics JSON found. Run the master pipeline to generate training history.")

with tab3:
    st.markdown("#### Project Evolution: Engineering & Audit Report")
    
    col_pca, col_strat = st.columns([1.2, 1], gap="large")
    
    with col_pca:
        st.markdown("#### Embedding Vector Space (PCA)")
        pca_data = get_pca_data()
        if pca_data:
            df_pca = pd.DataFrame(pca_data)
            fig_pca = px.scatter(df_pca, x="x", y="y", text="words", height=500)
            fig_pca.update_traces(textposition='top center', marker=dict(color="#7F77DD", size=8))
            fig_pca.update_layout(template="plotly_white", margin=dict(l=0,r=0,t=0,b=0), xaxis_visible=False, yaxis_visible=False)
            st.plotly_chart(fig_pca, use_container_width=True, config={'displayModeBar': False})
            st.markdown("<p class='muted-text'>Word relationships learned solely through sentiment backpropagation. Note how 'excellent' and 'great' drift away from 'worst'.</p>", unsafe_allow_html=True)
        else:
            st.info("Embedding visualization ready after training.")
            
        st.markdown("---")
        st.markdown("#### Project Story: Journey from Start to End")
        story = get_project_story()
        for item in story:
            with st.expander(item["stage"]):
                st.write(item["content"])

    with col_strat:
        st.markdown("#### Applied DL Strategies")
        strategies = [
            ("Custom Architecture", "A multi-layered LSTM architecture designed and trained specifically for sentiment analysis without relying on pre-trained weights."),
            ("Sequence Packing", "Packed padded sequences to bypass PAD tokens, optimizing LSTM memory bandwidth."),
            ("Gradient Clipping", "Hard thresholding at 1.0 to prevent exploding gradients during Backprop-Through-Time."),
            ("Feature Regularization", "Multi-layer dropout (0.3) and final linear dropout (0.5) to maintain generalization."),
            ("Sequential Analysis", "LSTM backbone captures long-term dependencies that Bag-of-Words models miss.")
        ]
        for title, desc in strategies:
            st.markdown(f"""
            <div class="strategy-card">
                <div style="font-size: 13px; font-weight: 600; color: #1a1a1a">{title}</div>
                <div style="font-size: 11px; color: #666; margin-top: 4px">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("---")
st.caption("Deep Learning Engineering Project · Sentiment LSTM Classifier v2.0")
