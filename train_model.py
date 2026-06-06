"""
Phishing Email Detection Model - Training Script
"""

import os
import re
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

DATASET_PATH = "dataset/phishing_emails.csv"
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "phishing_model.pkl")

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


def load_and_prepare_data():
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded {len(df)} records: {sum(df['label'] == 1)} phishing, {sum(df['label'] == 0)} safe")
    print("Preprocessing text...")
    df['clean_text'] = df['email_text'].apply(preprocess_text)
    print("Extracting URL features...")
    url_features_list = df['email_text'].apply(extract_url_features)
    url_df = pd.DataFrame(url_features_list.tolist())
    print("Extracting text features...")
    text_features_list = df['email_text'].apply(extract_text_features)
    text_features_df = pd.DataFrame(text_features_list.tolist())
    print("Combining features...")
    feature_df = pd.concat([url_df, text_features_df], axis=1)
    X_text = df['clean_text']
    X_features = feature_df
    y = df['label']
    return X_text, X_features, y


def train_models(X_text, X_features, y):
    print("\nSplitting data (80% train, 20% test)...")
    X_text_train, X_text_test, X_feat_train, X_feat_test, y_train, y_test = train_test_split(
        X_text, X_features, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {len(X_text_train)} samples")
    print(f"Testing set: {len(X_text_test)} samples")
    print("\nVectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True
    )
    X_text_train_tfidf = vectorizer.fit_transform(X_text_train)
    X_text_test_tfidf = vectorizer.transform(X_text_test)
    from scipy.sparse import hstack
    X_train = hstack([X_text_train_tfidf, X_feat_train.values])
    X_test = hstack([X_text_test_tfidf, X_feat_test.values])
    print(f"Combined feature matrix shape: {X_train.shape}")
    models = {}
    results = {}
    print("\nTraining Multinomial Naive Bayes...")
    nb_model = MultinomialNB(alpha=0.1)
    nb_model.fit(X_train, y_train)
    nb_pred = nb_model.predict(X_test)
    models['Naive Bayes'] = nb_model
    results['Naive Bayes'] = {
        'accuracy': accuracy_score(y_test, nb_pred),
        'precision': precision_score(y_test, nb_pred),
        'recall': recall_score(y_test, nb_pred),
        'f1': f1_score(y_test, nb_pred),
        'confusion_matrix': confusion_matrix(y_test, nb_pred),
        'classification_report': classification_report(y_test, nb_pred),
        'y_test': y_test,
        'y_pred': nb_pred
    }
    print(f"  Accuracy: {results['Naive Bayes']['accuracy']:.4f}")
    print(f"  F1-Score: {results['Naive Bayes']['f1']:.4f}")
    print("\nTraining Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    models['Random Forest'] = rf_model
    results['Random Forest'] = {
        'accuracy': accuracy_score(y_test, rf_pred),
        'precision': precision_score(y_test, rf_pred),
        'recall': recall_score(y_test, rf_pred),
        'f1': f1_score(y_test, rf_pred),
        'confusion_matrix': confusion_matrix(y_test, rf_pred),
        'classification_report': classification_report(y_test, rf_pred),
        'y_test': y_test,
        'y_pred': rf_pred
    }
    print(f"  Accuracy: {results['Random Forest']['accuracy']:.4f}")
    print(f"  F1-Score: {results['Random Forest']['f1']:.4f}")
    accuracies = {name: results[name]['accuracy'] for name in models}
    best_model_name = max(accuracies, key=accuracies.get)
    best_model = models[best_model_name]
    print(f"\n{'='*50}")
    print(f"Best Model: {best_model_name}")
    print(f"Best Accuracy: {accuracies[best_model_name]:.4f}")
    print(f"{'='*50}")
    return best_model, best_model_name, results, vectorizer, models, y_test


def plot_results(results):
    os.makedirs("screenshots", exist_ok=True)
    best_model_name = None
    best_acc = 0
    for name, res in results.items():
        if res['accuracy'] > best_acc:
            best_acc = res['accuracy']
            best_model_name = name
    cm = results[best_model_name]['confusion_matrix']
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Safe', 'Phishing'],
                yticklabels=['Safe', 'Phishing'],
                cbar_kws={'label': 'Count'},
                linewidths=0.5)
    plt.xlabel('Predicted Label', fontsize=11)
    plt.ylabel('Actual Label', fontsize=11)
    plt.title(f'Confusion Matrix - {best_model_name}', fontsize=13)
    plt.tight_layout()
    plt.savefig('screenshots/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nConfusion matrix saved to screenshots/confusion_matrix.png")
    model_names = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in model_names]
    precisions = [results[m]['precision'] for m in model_names]
    recalls = [results[m]['recall'] for m in model_names]
    f1_scores = [results[m]['f1'] for m in model_names]
    x = np.arange(len(model_names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 1.5*width, accuracies, width, label='Accuracy', color='#2ecc71')
    ax.bar(x - 0.5*width, precisions, width, label='Precision', color='#3498db')
    ax.bar(x + 0.5*width, recalls, width, label='Recall', color='#e74c3c')
    ax.bar(x + 1.5*width, f1_scores, width, label='F1-Score', color='#f39c12')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    for i, (acc, prec, rec, f1) in enumerate(zip(accuracies, precisions, recalls, f1_scores)):
        ax.text(i - 1.5*width, acc + 0.02, f"{acc:.3f}", ha='center', fontsize=8)
        ax.text(i - 0.5*width, prec + 0.02, f"{prec:.3f}", ha='center', fontsize=8)
        ax.text(i + 0.5*width, rec + 0.02, f"{rec:.3f}", ha='center', fontsize=8)
        ax.text(i + 1.5*width, f1 + 0.02, f"{f1:.3f}", ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig('screenshots/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Model comparison chart saved to screenshots/model_comparison.png")


def save_model_and_components(best_model, best_model_name, results, vectorizer, models):
    import json
    os.makedirs(MODELS_DIR, exist_ok=True)
    nb_res = results['Naive Bayes']
    rf_res = results['Random Forest']
    model_data = {
        'model': best_model,
        'vectorizer': vectorizer,
        'suspicious_keywords': SUSPICIOUS_KEYWORDS,
        'url_shorteners': URL_SHORTENERS,
        'suspicious_domains': SUSPICIOUS_DOMAINS,
        'best_model_name': best_model_name,
        'nb_accuracy': float(nb_res['accuracy']),
        'rf_accuracy': float(rf_res['accuracy']),
        'nb_precision': float(nb_res['precision']),
        'rf_precision': float(rf_res['precision']),
        'nb_recall': float(nb_res['recall']),
        'rf_recall': float(rf_res['recall']),
        'nb_f1': float(nb_res['f1']),
        'rf_f1': float(rf_res['f1']),
        'nb_cm': nb_res['confusion_matrix'].tolist(),
        'rf_cm': rf_res['confusion_matrix'].tolist(),
        'nb_report': nb_res['classification_report'],
        'rf_report': rf_res['classification_report'],
        'y_test': list(nb_res['y_test']),
        'nb_pred': list(nb_res['y_pred']),
        'rf_pred': list(rf_res['y_pred']),
        'results': {
            'Naive Bayes': {
                'accuracy': float(nb_res['accuracy']),
                'precision': float(nb_res['precision']),
                'recall': float(nb_res['recall']),
                'f1': float(nb_res['f1']),
                'confusion_matrix': nb_res['confusion_matrix'].tolist(),
                'classification_report': nb_res['classification_report']
            },
            'Random Forest': {
                'accuracy': float(rf_res['accuracy']),
                'precision': float(rf_res['precision']),
                'recall': float(rf_res['recall']),
                'f1': float(rf_res['f1']),
                'confusion_matrix': rf_res['confusion_matrix'].tolist(),
                'classification_report': rf_res['classification_report']
            }
        }
    }
    rf_model = models.get('Random Forest')
    if rf_model is not None and hasattr(rf_model, 'feature_importances_'):
        importances = rf_model.feature_importances_
        feature_names = vectorizer.get_feature_names_out().tolist() + [
            'url_count', 'has_ip_url', 'has_suspicious_domain',
            'has_shortener', 'url_keyword_score', 'exclamation_count',
            'urgent_word_count', 'link_word_count', 'text_length',
            'caps_ratio', 'dollar_count'
        ]
        feature_importance = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        model_data['feature_importance'] = [(f[0], float(f[1])) for f in feature_importance[:30]]
    else:
        model_data['feature_importance'] = []
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_data, f)
    metrics_path = os.path.join(MODELS_DIR, "model_metrics.json")
    metrics_for_json = {
        'best_model_name': best_model_name,
        'nb_accuracy': float(nb_res['accuracy']),
        'rf_accuracy': float(rf_res['accuracy']),
        'nb_precision': float(nb_res['precision']),
        'rf_precision': float(rf_res['precision']),
        'nb_recall': float(nb_res['recall']),
        'rf_recall': float(rf_res['recall']),
        'nb_f1': float(nb_res['f1']),
        'rf_f1': float(rf_res['f1']),
        'nb_cm': nb_res['confusion_matrix'].tolist(),
        'rf_cm': rf_res['confusion_matrix'].tolist(),
        'nb_report': nb_res['classification_report'],
        'rf_report': rf_res['classification_report'],
        'results': {
            'Naive Bayes': {
                'accuracy': float(nb_res['accuracy']),
                'precision': float(nb_res['precision']),
                'recall': float(nb_res['recall']),
                'f1': float(nb_res['f1']),
                'confusion_matrix': nb_res['confusion_matrix'].tolist(),
                'classification_report': nb_res['classification_report']
            },
            'Random Forest': {
                'accuracy': float(rf_res['accuracy']),
                'precision': float(rf_res['precision']),
                'recall': float(rf_res['recall']),
                'f1': float(rf_res['f1']),
                'confusion_matrix': rf_res['confusion_matrix'].tolist(),
                'classification_report': rf_res['classification_report']
            }
        },
        'feature_importance': model_data.get('feature_importance', [])
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_for_json, f, indent=2)
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Metrics saved to {metrics_path}")


def print_evaluation_results(results):
    print("\n" + "="*60)
    print("FULL EVALUATION RESULTS")
    print("="*60)
    for model_name, res in results.items():
        print(f"\n--- {model_name} ---")
        print(f"Accuracy:  {res['accuracy']:.4f}")
        print(f"Precision: {res['precision']:.4f}")
        print(f"Recall:    {res['recall']:.4f}")
        print(f"F1-Score:  {res['f1']:.4f}")
        print("\nClassification Report:")
        print(res['classification_report'])
        print(f"Confusion Matrix:\n{res['confusion_matrix']}")
    print("="*60)


def main():
    print("="*60)
    print("PHISHING EMAIL DETECTION MODEL - TRAINING PIPELINE")
    print("="*60)
    X_text, X_features, y = load_and_prepare_data()
    best_model, best_model_name, results, vectorizer, models, y_test = train_models(
        X_text, X_features, y
    )
    print_evaluation_results(results)
    plot_results(results)
    save_model_and_components(best_model, best_model_name, results, vectorizer, models)
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best model: {best_model_name}")
    print(f"Saved to: {MODEL_PATH}")
    print(f"Charts saved to: screenshots/")


if __name__ == "__main__":
    main()
