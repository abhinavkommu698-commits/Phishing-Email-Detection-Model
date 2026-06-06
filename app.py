"""
Phishing Email Detection System - Streamlit Web Application
"""

import streamlit as st
import pickle
import re
import os
import logging
from datetime import datetime

logging.basicConfig(
    filename='app_errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Phishing Email Detector",
    page_icon="🛡️",
    layout="centered"
)

MODEL_PATH = "models/phishing_model.pkl"
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

SUSPICIOUS_KEYWORDS = [
    'urgent', 'verify', 'account suspended', 'click here', 'login',
    'password reset', 'confirm', 'update', 'immediate', 'action required',
    'security alert', 'compromise', 'verify identity', 'restriction',
    'locked', 'blocked', 'unauthorized', 'suspicious activity',
    'failure', 'expire', 'expired', 'deactivation', 'reactivate',
    'temporary hold', 'payment', 'gift card', 'winner', 'claim now',
    'free money', 'award', 'refund', 'unclaimed', 'limited time',
    'restore access', 'confirm your identity', 'verify your account'
]

URL_SHORTENERS = [
    'bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'ow.ly', 'is.gd',
    'buff.ly', 'adf.ly', 'j.mp', 'bit.do', 'cutt.ly', 'shorturl.at'
]

SUSPICIOUS_DOMAINS = [
    'xyz', 'top', 'click', 'download', 'link', 'win', 'free',
    'gift', 'reward', 'bonus', 'prize', 'cash', 'loan', 'offer'
]

STOPWORDS = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'}


def simple_tokenize(text):
    return re.findall(r'\b[a-zA-Z]+\b', text)


def extract_url_features(text):
    features = {
        'url_count': 0,
        'has_ip_url': 0,
        'has_suspicious_domain': 0,
        'has_shortener': 0,
        'url_keyword_score': 0
    }
    url_pattern = r'https?://[^\s<>"\']+|www\.[^\s<>"\']+|[a-zA-Z0-9-]+\.(com|net|org|edu|gov|io|xyz|top|co|uk|ca|de|fr|in|au|ru|cn|jp|br|it|es|nl|se|no|fi|dk|pl|be|at|ch|pt|cz|hu|ro|gr|ie|nz|sk|bg|hr|lt|lv|si|ee|cy|mt|lux|is)[^\s<>"\']*'
    urls = re.findall(url_pattern, text, re.IGNORECASE)
    features['url_count'] = len(urls)
    ip_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    for url in urls:
        url_lower = url.lower()
        if re.search(ip_pattern, url):
            features['has_ip_url'] = 1
        for shortener in URL_SHORTENERS:
            if shortener in url_lower:
                features['has_shortener'] = 1
        domain_match = re.search(r'(?:https?://|www\.)?([^/\s]+)', url_lower)
        if domain_match:
            domain_parts = domain_match.group(1).split('.')
            for part in domain_parts:
                if part in SUSPICIOUS_DOMAINS:
                    features['has_suspicious_domain'] = 1
        for keyword in SUSPICIOUS_KEYWORDS:
            if keyword in url_lower:
                features['url_keyword_score'] += 1
    return features


def extract_text_features(text):
    features = {
        'exclamation_count': 0,
        'urgent_word_count': 0,
        'link_word_count': 0,
        'text_length': 0,
        'caps_ratio': 0.0,
        'dollar_count': 0
    }
    text_lower = text.lower()
    features['exclamation_count'] = text.count('!')
    features['text_length'] = len(text)
    features['dollar_count'] = text.count('$')
    total_chars = len([c for c in text if c.isalpha()])
    if total_chars > 0:
        upper_chars = sum(1 for c in text if c.isupper())
        features['caps_ratio'] = upper_chars / total_chars
    for keyword in SUSPICIOUS_KEYWORDS:
        features['urgent_word_count'] += text_lower.count(keyword)
    link_keywords = ['click here', 'click', 'link', 'url', 'website', 'page']
    for keyword in link_keywords:
        features['link_word_count'] += text_lower.count(keyword)
    return features


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://[^\s]+|www\.[^\s]+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = simple_tokenize(text)
    tokens = [word for word in tokens if word not in STOPWORDS]
    cleaned_text = ' '.join(tokens)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text


def load_model():
    try:
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        return None


def predict_email(email_text):
    model_data = load_model()
    if model_data is None:
        return None, None
    model = model_data['model']
    vectorizer = model_data['vectorizer']
    cleaned_text = preprocess_text(email_text)
    text_vectorized = vectorizer.transform([cleaned_text])
    url_features = extract_url_features(email_text)
    text_features = extract_text_features(email_text)
    import pandas as pd
    feature_df = pd.DataFrame([{**url_features, **text_features}])
    from scipy.sparse import hstack
    combined_features = hstack([text_vectorized, feature_df.values])
    prediction = model.predict(combined_features)[0]
    if hasattr(model, 'predict_proba'):
        confidence = max(model.predict_proba(combined_features)[0])
    else:
        confidence = 0.85
    return prediction, confidence


def detect_suspicious_urls(text):
    url_pattern = r'https?://[^\s<>"\']+|www\.[^\s<>"\']+'
    urls = re.findall(url_pattern, text, re.IGNORECASE)
    suspicious_findings = []
    ip_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    for url in urls:
        findings = []
        url_lower = url.lower()
        if re.search(ip_pattern, url):
            findings.append("IP address in URL")
        for shortener in URL_SHORTENERS:
            if shortener in url_lower:
                findings.append("URL shortener detected")
                break
        domain_match = re.search(r'(?:https?://|www\.)?([^/\s]+)', url_lower)
        if domain_match:
            domain_parts = domain_match.group(1).split('.')
            for part in domain_parts:
                if part in SUSPICIOUS_DOMAINS:
                    findings.append("Suspicious domain")
                    break
        for keyword in SUSPICIOUS_KEYWORDS:
            if keyword in url_lower:
                findings.append(f"Keyword: '{keyword}'")
                break
        if findings:
            suspicious_findings.append({'url': url, 'findings': findings})
    return suspicious_findings


def highlight_keywords(text, keywords):
    highlighted = text
    for keyword in sorted(keywords, key=len, reverse=True):
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        highlighted = pattern.sub(
            f'<span style="background-color: #ff4444; color: white; padding: 2px 4px; border-radius: 3px; font-weight: bold;">{keyword}</span>',
            highlighted
        )
    return highlighted


def calculate_risk_score(email_text, prediction, confidence):
    url_features = extract_url_features(email_text)
    text_features = extract_text_features(email_text)
    score = 0
    max_score = 100
    if prediction == 1:
        score += 40
    score += min(url_features['url_count'] * 10, 20)
    score += url_features['has_ip_url'] * 15
    score += url_features['has_shortener'] * 10
    score += url_features['has_suspicious_domain'] * 10
    score += min(url_features['url_keyword_score'] * 5, 15)
    score += min(text_features['urgent_word_count'] * 3, 15)
    score += text_features['exclamation_count'] * 2
    score += text_features['caps_ratio'] * 10
    score = min(score, max_score)
    score = round(score, 2)
    if score >= 80:
        risk_level = "CRITICAL"
        risk_color = "#dc3545"
    elif score >= 60:
        risk_level = "HIGH"
        risk_color = "#fd7e14"
    elif score >= 40:
        risk_level = "MEDIUM"
        risk_color = "#ffc107"
    elif score >= 20:
        risk_level = "LOW"
        risk_color = "#28a745"
    else:
        risk_level = "SAFE"
        risk_color = "#20c997"
    return min(score, 100), risk_level, risk_color


def sanitize_for_pdf(text):
    if text is None:
        return ""
    text = str(text)
    text = text.encode('latin-1', 'replace').decode('latin-1')
    text = text.replace('\x00', '')
    return text


def generate_pdf_report(email_text, prediction, confidence, risk_score, risk_level, suspicious_urls, model_data):
    try:
        from fpdf import FPDF
        if email_text is None or not str(email_text).strip():
            raise ValueError("Email text is empty")
        if model_data is None:
            raise ValueError("Model data is not available")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 20)
        pdf.cell(0, 15, "Phishing Email Analysis Report", ln=True, align='C')
        pdf.ln(5)
        pdf.set_font("Arial", '', 10)
        report_time = datetime.now().strftime('%B %d, %Y at %I:%M %p')
        pdf.cell(0, 8, f"Report Generated: {report_time}", ln=True, align='C')
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Analysis Result", ln=True)
        pdf.set_font("Arial", '', 11)
        if prediction == 1:
            pdf.set_text_color(220, 53, 69)
            pdf.cell(0, 8, "RESULT: PHISHING EMAIL DETECTED", ln=True)
        else:
            pdf.set_text_color(40, 167, 69)
            pdf.cell(0, 8, "RESULT: SAFE EMAIL", ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 8, f"Confidence: {confidence*100:.1f}%", ln=True)
        pdf.cell(0, 8, f"Risk Score: {risk_score:.2f}/100 ({sanitize_for_pdf(risk_level)})", ln=True)
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Email Content", ln=True)
        pdf.set_font("Arial", '', 10)
        safe_text = sanitize_for_pdf(str(email_text))
        pdf.multi_cell(0, 6, safe_text)
        pdf.ln(3)
        if suspicious_urls:
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "Suspicious URLs Detected", ln=True)
            pdf.set_font("Arial", '', 10)
            for item in suspicious_urls:
                safe_url = sanitize_for_pdf(item.get('url', ''))[:80]
                findings = [sanitize_for_pdf(f) for f in item.get('findings', [])]
                pdf.set_text_color(220, 53, 69)
                pdf.cell(0, 6, f"URL: {safe_url}", ln=True)
                pdf.set_text_color(0, 0, 0)
                pdf.cell(0, 6, f"  Findings: {', '.join(findings)}", ln=True)
            pdf.ln(3)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Model Information", ln=True)
        pdf.set_font("Arial", '', 10)
        pdf.cell(0, 6, f"Best Model: {sanitize_for_pdf(model_data.get('best_model_name', 'N/A'))}", ln=True)
        nb_acc = model_data.get('nb_accuracy', 0)
        rf_acc = model_data.get('rf_accuracy', 0)
        if nb_acc is not None:
            pdf.cell(0, 6, f"Naive Bayes Accuracy: {nb_acc*100:.1f}%", ln=True)
        if rf_acc is not None:
            pdf.cell(0, 6, f"Random Forest Accuracy: {rf_acc*100:.1f}%", ln=True)
        os.makedirs(REPORTS_DIR, exist_ok=True)
        report_path = os.path.join(REPORTS_DIR, f"phishing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
        pdf.output(report_path)
        if not os.path.exists(report_path):
            raise RuntimeError("PDF file was not created")
        return report_path
    except Exception as e:
        logger.error(f"PDF generation failed: {e}", exc_info=True)
        return None


def apply_dark_theme():
    st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #252540 100%);
        border: 1px solid #3a3a5c;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-3px);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #64ffda;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .safe-result {
        background: linear-gradient(135deg, #0d3d22 0%, #134e2e 100%);
        border: 2px solid #28a745;
    }
    .phishing-result {
        background: linear-gradient(135deg, #4a1a1a 0%, #5c2424 100%);
        border: 2px solid #dc3545;
    }
    .section-header {
        color: #e0e0e0;
        font-weight: 600;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3a3a5c;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #64ffda 0%, #4ec9b0 100%);
        color: #0e1117;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #4ec9b0 0%, #3db89a 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(100, 255, 218, 0.3);
    }
    .stTextArea textarea {
        background-color: #1e1e2f;
        color: #fafafa;
        border: 2px solid #3a3a5c;
        border-radius: 10px;
    }
    .stTextArea textarea:focus {
        border-color: #64ffda;
        box-shadow: 0 0 0 2px rgba(100, 255, 218, 0.2);
    }
    .highlight-keyword {
        background-color: #ff4444;
        color: white;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: bold;
    }
    .url-item {
        background-color: #1e1e2f;
        border-left: 4px solid #dc3545;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.85rem;
        word-break: break-all;
    }
    .finding-tag {
        display: inline-block;
        background-color: #dc3545;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        margin: 2px 4px 2px 0;
    }
    </style>
    """, unsafe_allow_html=True)


def main():
    apply_dark_theme()
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("Phishing Email Detection System")
    st.markdown("*Analyze emails for phishing threats using advanced ML*")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Features Analyzed", "10+", help="URL patterns, keywords, metadata")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Models Compared", "2", help="Naive Bayes & Random Forest")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Algorithm", "TF-IDF + ML", help="Text vectorization with ML classifiers")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="section-header">Email Analysis</div>', unsafe_allow_html=True)
    with st.expander("Try Sample Emails"):
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Example Phishing Email"):
                st.session_state.sample_email = "Urgent: Your account has been suspended. Click here to verify your identity: http://192.168.1.1/login.html and restore access immediately. Failure to act will result in permanent account deletion."
        with col_b:
            if st.button("Example Safe Email"):
                st.session_state.sample_email = "Hi John, just wanted to follow up on our meeting scheduled for tomorrow at 10 AM in the conference room. Please let me know if you need any additional materials prepared."
    email_input = st.text_area(
        "Paste your email content here:",
        height=180,
        placeholder="Enter or paste the email text you want to analyze...",
        value=st.session_state.get('sample_email', '')
    )
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        check_button = st.button("Check Email", use_container_width=True, type="primary")
    if check_button or email_input != '':
        if not email_input.strip():
            st.warning("Please enter an email to analyze.")
        else:
            with st.spinner("Analyzing email..."):
                prediction, confidence = predict_email(email_input)
                risk_score, risk_level, risk_color = calculate_risk_score(email_input, prediction, confidence)
                suspicious_urls = detect_suspicious_urls(email_input)
                if prediction is None:
                    st.error("Model not found! Please run `python3 train_model.py` first.")
                    st.code("python3 train_model.py", language="bash")
                else:
                    if prediction == 1:
                        st.markdown(f"""
                        <div class="result-box phishing-result">
                            <h2 style="color: #ff6b6b;">WARNING: Phishing Email Detected</h2>
                            <p style="font-size: 1.1rem; color: #ffcccc;">This email shows characteristics of a phishing attempt.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-box safe-result">
                            <h2 style="color: #64ffda;">Safe Email</h2>
                            <p style="font-size: 1.1rem; color: #d4edda;">This email appears to be legitimate and safe.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown("---")
                    st.markdown('<div class="section-header">Analysis Results</div>', unsafe_allow_html=True)
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Prediction", "PHISHING" if prediction == 1 else "SAFE", delta="Suspicious" if prediction == 1 else "Clean")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Confidence", f"{confidence*100:.1f}%", delta="High" if confidence > 0.8 else "Medium")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col3:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Risk Score", f"{risk_score:.2f}/100", delta=risk_level)
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col4:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        progress_val = risk_score / 100
                        st.markdown(f'<div class="metric-value" style="color: {risk_color};">{risk_score:.2f}</div>', unsafe_allow_html=True)
                        st.progress(progress_val)
                        st.caption("Risk Level")
                        st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown('<div class="section-header">Suspicious Keywords</div>', unsafe_allow_html=True)
                        found_keywords = []
                        text_lower = email_input.lower()
                        for keyword in SUSPICIOUS_KEYWORDS:
                            if keyword in text_lower:
                                found_keywords.append(keyword)
                        if found_keywords:
                            highlighted_text = highlight_keywords(email_input, SUSPICIOUS_KEYWORDS)
                            st.markdown(f'<div style="background-color: #1a1a2e; padding: 1rem; border-radius: 10px; border: 1px solid #3a3a5c; line-height: 1.6;">{highlighted_text}</div>', unsafe_allow_html=True)
                            st.caption(f"Found {len(found_keywords)} suspicious keyword(s): {', '.join(found_keywords[:5])}")
                        else:
                            st.info("No suspicious keywords detected.")
                    with col2:
                        st.markdown('<div class="section-header">URL Analysis</div>', unsafe_allow_html=True)
                        url_features = extract_url_features(email_input)
                        st.metric("Total URLs Found", url_features['url_count'])
                        if suspicious_urls:
                            st.caption(f"Found {len(suspicious_urls)} suspicious URL(s):")
                            for item in suspicious_urls:
                                st.markdown(f"""
                                <div class="url-item">
                                    <strong>URL:</strong> {item['url'][:70]}{'...' if len(item['url']) > 70 else ''}<br>
                                    <strong>Findings:</strong><br>
                                    {' '.join([f'<span class="finding-tag">{f}</span>' for f in item['findings']])}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            if url_features['url_count'] > 0:
                                st.info(f"Found {url_features['url_count']} URL(s) but none appear suspicious.")
                            else:
                                st.info("No URLs detected in this email.")
                    st.markdown("---")
                    model_data = load_model()
                    if model_data:
                        report_path = generate_pdf_report(
                            email_text=email_input,
                            prediction=prediction,
                            confidence=confidence,
                            risk_score=risk_score,
                            risk_level=risk_level,
                            suspicious_urls=suspicious_urls,
                            model_data=model_data
                        )
                        if report_path and os.path.exists(report_path):
                            try:
                                with open(report_path, 'rb') as f:
                                    pdf_bytes = f.read()
                                st.download_button(
                                    label="Download PDF Report",
                                    data=pdf_bytes,
                                    file_name=os.path.basename(report_path),
                                    mime="application/pdf",
                                    use_container_width=True
                                )
                            except Exception as e:
                                logger.error(f"Failed to read PDF for download: {e}")
                                st.warning("PDF report generated but could not be downloaded. Please try again.")
                        else:
                            logger.warning("PDF generation returned None or file does not exist")
                            st.warning("PDF report generation failed. The analysis is still complete — you can view all results above.")
    st.markdown("---")
    st.markdown('<div class="section-header">How It Works</div>', unsafe_allow_html=True)
    with st.expander("1. Text Preprocessing"):
        st.write("""
        - Converts text to lowercase
        - Removes URLs, punctuation, and special characters
        - Removes stopwords using NLTK
        - Cleans and normalizes the content
        """)
    with st.expander("2. Feature Extraction"):
        st.write("""
        **Text Features (TF-IDF):**
        - Bag-of-words representation
        - N-grams (1-gram and 2-gram)
        - 5000 most frequent terms
        
        **URL Features:**
        - Total URL count
        - IP addresses in URLs
        - URL shortening services detected
        - Suspicious domain patterns
        """)
    with st.expander("3. Risk Score Calculation"):
        st.write("""
        The risk score (0-100) is calculated based on:
        - Model prediction confidence
        - Number of URLs in the email
        - Presence of IP addresses in URLs
        - URL shortener usage
        - Suspicious domain patterns
        - Urgent keyword frequency
        - Excessive exclamation marks
        - Unusual capitalization patterns
        """)
    st.markdown("---")
    st.markdown("<center><small>Phishing Email Detection Model | Built with Streamlit & Scikit-learn</small></center>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
