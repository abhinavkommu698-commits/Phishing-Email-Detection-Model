"""
Dataset Preview Page - Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(
    page_title="Dataset Preview",
    page_icon="📊",
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
.stat-card {
    background: linear-gradient(135deg, #1e1e2f 0%, #252540 100%);
    border: 1px solid #3a3a5c;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
}
.stat-value {
    font-size: 1.8rem;
    font-weight: bold;
    color: #64ffda;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("Dataset Preview")
st.markdown("*Explore the phishing email dataset used for model training*")
st.markdown('</div>', unsafe_allow_html=True)

DATASET_PATH = "dataset/phishing_emails.csv"

if not os.path.exists(DATASET_PATH):
    st.error("Dataset not found. Please run `python3 generate_dataset.py` to create the dataset.")
else:
    df = pd.read_csv(DATASET_PATH)
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown('<div class="stat-value">1000</div>', unsafe_allow_html=True)
        st.caption("Total Records")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        phishing_count = (df['label'] == 1).sum()
        st.markdown(f'<div class="stat-value">{phishing_count}</div>', unsafe_allow_html=True)
        st.caption("Phishing Emails")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        safe_count = (df['label'] == 0).sum()
        st.markdown(f'<div class="stat-value">{safe_count}</div>', unsafe_allow_html=True)
        st.caption("Safe Emails")
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        avg_len = df['email_text'].str.len().mean()
        st.markdown(f'<div class="stat-value">{avg_len:.0f}</div>', unsafe_allow_html=True)
        st.caption("Avg Length (chars)")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Label Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        counts = df['label'].value_counts()
        labels = ['Safe (0)', 'Phishing (1)']
        colors = ['#28a745', '#dc3545']
        bars = ax.bar(labels, [counts.get(0, 0), counts.get(1, 0)], color=colors, width=0.5, edgecolor='#3a3a5c', linewidth=1.5)
        ax.set_ylabel('Count', color='#e0e0e0')
        ax.set_title('Email Classification', color='#e0e0e0', pad=10)
        ax.tick_params(colors='#a0a0b0')
        ax.spines['bottom'].set_color('#3a3a5c')
        ax.spines['left'].set_color('#3a3a5c')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#1a1a2e')
        ax.figure.patch.set_facecolor('#1a1a2e')
        ax.grid(True, alpha=0.2, axis='y', color='#3a3a5c')
        for bar, count in zip(bars, [counts.get(0, 0), counts.get(1, 0)]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(count), ha='center', va='bottom', fontsize=12, color='#64ffda', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    with col2:
        st.markdown("### Email Length Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        phishing_lengths = df[df['label'] == 1]['email_text'].str.len()
        safe_lengths = df[df['label'] == 0]['email_text'].str.len()
        ax.hist(safe_lengths, bins=20, alpha=0.7, label='Safe', color='#28a745', edgecolor='#3a3a5c')
        ax.hist(phishing_lengths, bins=20, alpha=0.7, label='Phishing', color='#dc3545', edgecolor='#3a3a5c')
        ax.set_xlabel('Length (characters)', color='#e0e0e0')
        ax.set_ylabel('Frequency', color='#e0e0e0')
        ax.set_title('Email Length by Class', color='#e0e0e0', pad=10)
        ax.tick_params(colors='#a0a0b0')
        ax.spines['bottom'].set_color('#3a3a5c')
        ax.spines['left'].set_color('#3a3a5c')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#1a1a2e')
        ax.figure.patch.set_facecolor('#1a1a2e')
        ax.grid(True, alpha=0.2, axis='y', color='#3a3a5c')
        ax.legend(facecolor='#1e1e2f', labelcolor='#e0e0e0')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    st.markdown("---")
    st.markdown("### Sample Records")
    tab1, tab2 = st.tabs(["Phishing Emails", "Safe Emails"])
    with tab1:
        phishing_samples = df[df['label'] == 1].head(10)[['email_text', 'label']]
        st.dataframe(
            phishing_samples.style.set_properties(**{
                'background-color': '#2d1f1f',
                'color': '#ffcccc',
                'border-color': '#5c2424'
            }),
            use_container_width=True,
            hide_index=True
        )
    with tab2:
        safe_samples = df[df['label'] == 0].head(10)[['email_text', 'label']]
        st.dataframe(
            safe_samples.style.set_properties(**{
                'background-color': '#1f2d1f',
                'color': '#ccffcc',
                'border-color': '#245c24'
            }),
            use_container_width=True,
            hide_index=True
        )
    st.markdown("---")
    st.markdown("<center><small>Phishing Email Detection Model | Dataset Analytics</small></center>", unsafe_allow_html=True)
