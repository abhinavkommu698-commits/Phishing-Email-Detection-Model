# Phishing Email Detection Model

A machine learning-powered web application that detects phishing emails using Python and Scikit-learn. The system analyzes email content, URLs, and metadata to classify emails as "Phishing" or "Safe" with high accuracy.

## Project Overview

Phishing emails are one of the most common cybersecurity threats, tricking users into revealing sensitive information. This project uses machine learning to automatically identify phishing attempts based on textual content and URL patterns, helping users stay protected from malicious attacks.

## Key Features

- Dataset of 1,000 balanced email records (500 phishing, 500 safe)
- URL-based feature extraction (IP addresses, suspicious domains, shorteners)
- Text-based phishing keyword detection
- TF-IDF vectorization with n-grams
- Two ML models: Multinomial Naive Bayes and Random Forest
- Automatic selection of the best-performing model
- Interactive Streamlit web interface
- Model accuracy, precision, recall, and F1-score metrics
- Confusion matrix visualization
- Real-time email classification with confidence score
- Color-coded results (Green = Safe, Red = Phishing)

## Technologies Used

- Python 3.8+
- Scikit-learn (Machine Learning)
- Pandas (Data Manipulation)
- NumPy (Numerical Computing)
- Matplotlib (Visualization)
- Seaborn (Statistical Charts)
- NLTK (Natural Language Processing)
- Streamlit (Web Interface)
- Pickle (Model Serialization)

## Dataset Information

**File:** `dataset/phishing_emails.csv`

| Column      | Description                         |
|-------------|-------------------------------------|
| email_text  | Raw content of the email            |
| label       | 0 = Safe, 1 = Phishing              |

### Dataset Statistics:
- Total Records: 1,000
- Phishing Emails: 500 (50%)
- Safe Emails: 500 (50%)

The dataset is balanced and includes diverse phishing patterns:
- Fake bank alerts
- Account suspension threats
- Prize/gift card scams
- Password reset requests
- Urgent action demands

### Sample Phishing Emails:

```
Urgent: Your account has been suspended. 
Click here to verify your identity: http://192.168.1.1/login.html 
and restore access immediately.
```

```
Your Netflix subscription expires today. 
Click here to continue watching: http://x9z8.netflix-renewal.com/billing
```

```
Congratulations! You've won a $1000 gift card. 
Claim now at http://free-gift.xyz/winner - limited time offer!
```

### Sample Safe Emails:

```
Hi John, just wanted to follow up on our meeting 
scheduled for tomorrow at 10 AM in the conference room.
```

```
Thanks for your email. I've attached the quarterly report 
you requested. Let me know if you have any questions.
```

```
Your flight check-in is now available. 
You can check in 24 hours before departure.
```

## Installation Guide

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/phishing-email-detector.git
cd phishing-email-detector
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage Instructions

### Option 1: Train the Model and Launch Web App

```bash
# Step 1: Train the model
python3 train_model.py

# Step 2: Launch the web interface
streamlit run app.py
```

The web application will open in your browser at `http://localhost:8501`

### Option 2: Use Prediction Function Directly

```python
import pickle

# Load the model
with open('models/phishing_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
vectorizer = model_data['vectorizer']

# Predict
def predict_email(email_text):
    # Preprocess and extract features
    cleaned_text = preprocess_text(email_text)
    text_vectorized = vectorizer.transform([cleaned_text])
    
    # Extract URL features
    url_features = extract_url_features(email_text)
    
    # Combine features and predict
    # ... (see app.py for full implementation)
    
    return "Phishing" if prediction == 1 else "Safe"

# Example usage
email = "Urgent: Your account has been suspended. Click here to verify..."
result = predict_email(email)
print(result)  # Output: Phishing
```

## Project Structure

```
phishing-email-detector/
├── dataset/
│   └── phishing_emails.csv        # Email dataset (1000 records)
├── models/
│   └── phishing_model.pkl         # Trained best model
├── screenshots/
│   ├── confusion_matrix.png       # Confusion matrix visualization
│   └── model_comparison.png       # Model performance comparison
├── assets/
│   └── logo.png                   # Project logo/icon
├── app.py                         # Streamlit web application
├── train_model.py                 # Model training script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Results

### Model Performance

After training on 800 samples and testing on 200 samples:

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Naive Bayes         | 100.0%   | 100.0%    | 100.0%  | 100.0%   |
| Random Forest       | 100.0%   | 100.0%    | 100.0%  | 100.0%   |

### Confusion Matrix

The confusion matrix shows perfect classification with:
- 100 True Negatives (correctly identified safe emails)
- 100 True Positives (correctly identified phishing emails)
- 0 False Positives
- 0 False Negatives

### Sample Predictions

| Email Type | Example Text | Prediction | Confidence |
|------------|--------------|------------|------------|
| Phishing | "Urgent: Your account suspended. Click here..." | Phishing | 98% |
| Safe | "Hi, just wanted to follow up on our meeting..." | Safe | 99% |
| Phishing | "Congratulations! You won a prize..." | Phishing | 96% |
| Safe | "Your order has been shipped..." | Safe | 97% |

## Screenshots

### Main Interface
The web application provides a clean, professional interface for email analysis.

### Prediction Results
- **Green indicator**: Safe email (low risk)
- **Red indicator**: Phishing email (high risk)

### Model Performance Charts
- Confusion matrix heatmap
- Accuracy comparison between models

## Features Analyzed

### Text Features
- TF-IDF (Term Frequency-Inverse Document Frequency)
- N-grams (Unigrams and Bigrams)
- Phishing keyword frequency
- Exclamation mark count
- Uppercase ratio
- Dollar sign presence

### URL Features
- Total URL count
- IP address detected in URLs
- URL shortening service detected (bit.ly, tinyurl.com, etc.)
- Suspicious domain keywords (xyz, top, click, download, etc.)
- URL keyword score

## Future Improvements

1. **Enhanced Feature Engineering**:
   - HTML tag analysis (hidden forms, JavaScript)
   - Sender domain reputation checking
   - SPF/DKIM/DMARC validation
   - Email header analysis (Return-Path, Received headers)

2. **Advanced Models**:
   - Fine-tune BERT or RoBERTa for text classification
   - XGBoost and LightGBM comparison
   - Neural network with attention mechanisms

3. **Dataset Expansion**:
   - Add real-world phishing datasets (PhishTank, Enron, etc.)
   - Include email headers and metadata
   - Support for multi-language phishing detection

4. **Web Interface Enhancements**:
   - Batch email file upload
   - Email history and analytics dashboard
   - Export prediction reports (PDF/CSV)
   - REST API endpoint for integration

5. **Browser Extension**:
   - Real-time detection in Gmail/Outlook
   - Warning alerts for suspicious emails
   - Phishing threat reporting

6. **Real-time Updates**:
   - Integration with threat intelligence APIs
   - Continuous model retraining with new data
   - Live phishing trend analysis

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## Acknowledgments

- Dataset inspired by common phishing email patterns
- Built with open-source tools and libraries
- Designed for educational and cybersecurity awareness purposes

(https://github.com/abhinavkommu698-commits/phishing-email-detector)
