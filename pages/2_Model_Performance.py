"""
Model Performance Dashboard - Streamlit
Displays comprehensive model evaluation metrics and visualizations.
"""

import streamlit as st
import pickle
import json
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

logging.basicConfig(
    filename='dashboard_errors.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Model Performance",
    page_icon="📈",
    layout="centered"
)

st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 1.5rem 0;
    background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
    border-radius: 12px;
    margin-bottom: 1.5rem;
}
.section-header {
    color: #e0e0e0;
    font-weight: 600;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #3a3a5c;
    margin-bottom: 1rem;
}
.metric-card {
    background: linear-gradient(135deg, #1e1e2f 0%, #252540 100%);
    border: 1px solid #3a3a5c;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
}
.metric-value {
    font-size: 1.6rem;
    font-weight: bold;
    color: #64ffda;
}
.chart-container {
    background: #1a1a2e;
    border-radius: 12px;
    padding: 1rem;
    border: 1px solid #3a3a5c;
}
.report-container {
    background: #1a1a2e;
    border-radius: 12px;
    padding: 1.2rem;
    border: 1px solid #3a3a5c;
    font-family: 'Courier New', monospace;
    font-size: 0.85rem;
    color: #d0d0d0;
    overflow-x: auto;
    white-space: pre;
}
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.json")
MODEL_PATH = os.path.join(MODELS_DIR, "phishing_model.pkl")

logger.info(f"Dashboard base directory: {BASE_DIR}")
logger.info(f"Looking for metrics at: {METRICS_PATH}")


@st.cache_resource
def load_metrics():
    try:
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, 'r') as f:
                data = json.load(f)
            logger.info("Successfully loaded model_metrics.json")
            return data
        else:
            logger.warning(f"model_metrics.json not found at {METRICS_PATH}")
            return None
    except Exception as e:
        logger.error(f"Error loading model_metrics.json: {e}", exc_info=True)
        return None


@st.cache_resource
def load_model_data():
    try:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                return pickle.load(f)
        else:
            logger.warning(f"phishing_model.pkl not found at {MODEL_PATH}")
            return None
    except Exception as e:
        logger.error(f"Error loading phishing_model.pkl: {e}", exc_info=True)
        return None


st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("Model Performance Dashboard")
st.markdown("*Comprehensive evaluation of phishing detection models*")
st.markdown('</div>', unsafe_allow_html=True)

metrics_data = load_metrics()
model_data = load_model_data()

if metrics_data is None:
    st.warning("Model metrics file not found.")
    st.info("Please run `python3 train_model.py` to train the model and generate performance metrics.")
    st.caption(f"Expected file location: `{METRICS_PATH}`")
else:
    nb_res = metrics_data['results']['Naive Bayes']
    rf_res = metrics_data['results']['Random Forest']
    best_model_name = metrics_data.get('best_model_name', 'Naive Bayes')
    st.markdown('<div class="section-header">Performance Summary</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Multinomial Naive Bayes")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{nb_res["accuracy"]:.1%}</div>', unsafe_allow_html=True)
            st.caption("Accuracy")
            st.markdown('</div>', unsafe_allow_html=True)
        with m2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{nb_res["precision"]:.1%}</div>', unsafe_allow_html=True)
            st.caption("Precision")
            st.markdown('</div>', unsafe_allow_html=True)
        with m3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{nb_res["recall"]:.1%}</div>', unsafe_allow_html=True)
            st.caption("Recall")
            st.markdown('</div>', unsafe_allow_html=True)
        with m4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{nb_res["f1"]:.1%}</div>', unsafe_allow_html=True)
            st.caption("F1-Score")
            st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("### Random Forest Classifier")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{rf_res["accuracy"]:.1%}</div>', unsafe_allow_html=True)
            st.caption("Accuracy")
            st.markdown('</div>', unsafe_allow_html=True)
        with m2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{rf_res["precision"]:.1%}</div>', unsafe_allow_html=True)
            st.caption("Precision")
            st.markdown('</div>', unsafe_allow_html=True)
        with m3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{rf_res["recall"]:.1%}</div>', unsafe_allow_html=True)
            st.caption("Recall")
            st.markdown('</div>', unsafe_allow_html=True)
        with m4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{rf_res["f1"]:.1%}</div>', unsafe_allow_html=True)
            st.caption("F1-Score")
            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="section-header">Best Model</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Best Model", best_model_name)
    with col2:
        if best_model_name == "Naive Bayes":
            st.metric("Best Accuracy", f"{nb_res['accuracy']:.1%}")
        else:
            st.metric("Best Accuracy", f"{rf_res['accuracy']:.1%}")
    with col3:
        if best_model_name == "Naive Bayes":
            st.metric("Best F1-Score", f"{nb_res['f1']:.1%}")
        else:
            st.metric("Best F1-Score", f"{rf_res['f1']:.1%}")
    with col4:
        st.metric("Models Trained", "2")
    st.markdown("---")
    st.markdown('<div class="section-header">Visual Analysis</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Confusion Matrix")
        st.caption(f"Best Model: {best_model_name}")
        try:
            cm = np.array(metrics_data['results'][best_model_name]['confusion_matrix'])
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn',
                        xticklabels=['Safe', 'Phishing'],
                        yticklabels=['Safe', 'Phishing'],
                        cbar_kws={'label': 'Count'},
                        linewidths=0.5,
                        linecolor='#3a3a5c',
                        ax=ax,
                        vmin=0, vmax=cm.max())
            ax.set_xlabel('Predicted Label', fontsize=11, color='#e0e0e0')
            ax.set_ylabel('Actual Label', fontsize=11, color='#e0e0e0')
            ax.set_title(f'{best_model_name}', fontsize=13, color='#e0e0e0', pad=10)
            ax.tick_params(colors='#a0a0b0')
            ax.figure.patch.set_facecolor('#1a1a2e')
            ax.set_facecolor('#1a1a2e')
            cbar = ax.collections[0].colorbar
            cbar.ax.yaxis.label.set_color('#a0a0b0')
            cbar.ax.tick_params(colors='#a0a0b0')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {e}", exc_info=True)
            st.error("Failed to render confusion matrix.")
    with col2:
        st.markdown("### Accuracy Comparison")
        fig, ax = plt.subplots(figsize=(6, 5))
        model_names = ['Naive Bayes', 'Random Forest']
        accuracies = [nb_res['accuracy'], rf_res['accuracy']]
        colors = ['#64ffda', '#ff6b6b']
        bars = ax.bar(model_names, accuracies, color=colors, width=0.5, edgecolor='#3a3a5c', linewidth=1.5)
        ax.set_ylabel('Accuracy', fontsize=12, color='#e0e0e0')
        ax.set_title('Classifier Accuracy', fontsize=13, color='#e0e0e0', pad=10)
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], color='#a0a0b0')
        ax.tick_params(colors='#a0a0b0')
        ax.spines['bottom'].set_color('#3a3a5c')
        ax.spines['left'].set_color('#3a3a5c')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#1a1a2e')
        ax.figure.patch.set_facecolor('#1a1a2e')
        ax.grid(True, alpha=0.2, axis='y', color='#3a3a5c')
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{acc:.1%}', ha='center', va='bottom', fontsize=12, color='#64ffda', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Multi-Metric Comparison")
        x = np.arange(4)
        width = 0.35
        fig, ax = plt.subplots(figsize=(6, 4))
        nb_vals = [nb_res['accuracy'], nb_res['precision'], nb_res['recall'], nb_res['f1']]
        rf_vals = [rf_res['accuracy'], rf_res['precision'], rf_res['recall'], rf_res['f1']]
        ax.bar(x - width/2, nb_vals, width, label='Naive Bayes', color='#64ffda', edgecolor='#3a3a5c')
        ax.bar(x + width/2, rf_vals, width, label='Random Forest', color='#ff6b6b', edgecolor='#3a3a5c')
        ax.set_ylabel('Score', fontsize=12, color='#e0e0e0')
        ax.set_title('All Metrics Comparison', fontsize=13, color='#e0e0e0', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score'], color='#a0a0b0')
        ax.set_ylim(0, 1.1)
        ax.tick_params(colors='#a0a0b0')
        ax.spines['bottom'].set_color('#3a3a5c')
        ax.spines['left'].set_color('#3a3a5c')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#1a1a2e')
        ax.figure.patch.set_facecolor('#1a1a2e')
        ax.grid(True, alpha=0.2, axis='y', color='#3a3a5c')
        ax.legend(facecolor='#1e1e2f', labelcolor='#e0e0e0')
        for i, (nb, rf) in enumerate(zip(nb_vals, rf_vals)):
            ax.text(i - width/2, nb + 0.02, f'{nb:.2f}', ha='center', fontsize=8, color='#64ffda')
            ax.text(i + width/2, rf + 0.02, f'{rf:.2f}', ha='center', fontsize=8, color='#ff6b6b')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    with col2:
        st.markdown("### Feature Importance (Top 15)")
        feature_importance = None
        if model_data is not None and 'feature_importance' in model_data:
            feature_importance = model_data['feature_importance']
        elif 'feature_importance' in metrics_data:
            feature_importance = metrics_data['feature_importance']
        if feature_importance:
            top_features = feature_importance[:15]
            features = [f[0] for f in top_features]
            importances = [float(f[1]) for f in top_features]
            fig, ax = plt.subplots(figsize=(6, 6))
            y_pos = np.arange(len(features))
            bars = ax.barh(y_pos, importances, color='#64ffda', edgecolor='#3a3a5c', height=0.6)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, color='#e0e0e0', fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel('Importance', color='#e0e0e0')
            ax.set_title('Random Forest Feature Importance', fontsize=13, color='#e0e0e0', pad=10)
            ax.tick_params(colors='#a0a0b0')
            ax.spines['bottom'].set_color('#3a3a5c')
            ax.spines['left'].set_color('#3a3a5c')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_facecolor('#1a1a2e')
            ax.figure.patch.set_facecolor('#1a1a2e')
            ax.grid(True, alpha=0.2, axis='x', color='#3a3a5c')
            for i, (bar, imp) in enumerate(zip(bars, importances)):
                ax.text(imp + 0.005, i, f'{imp:.3f}', va='center', fontsize=8, color='#64ffda')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Feature importance data not available.")
    st.markdown("---")
    with st.expander("View Classification Report (Naive Bayes)"):
        st.markdown('<div class="report-container">', unsafe_allow_html=True)
        st.text(nb_res['classification_report'])
        st.markdown('</div>', unsafe_allow_html=True)
    with st.expander("View Classification Report (Random Forest)"):
        st.markdown('<div class="report-container">', unsafe_allow_html=True)
        st.text(rf_res['classification_report'])
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<center><small>Phishing Email Detection Model | Performance Analytics</small></center>", unsafe_allow_html=True)
