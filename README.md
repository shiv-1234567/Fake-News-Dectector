# Fake-News-Dectector


This Fake News Detection app uses Logistic Regression with TF-IDF features to classify news articles as Fake or Real with 98% accuracy. Built using Streamlit, it processes political and global news inputs, displaying predictions with confidence levels, a styled preview, and explanations. Ideal for verifying potentially misleading news content.

📰 Fake News Detection System
🔍 Detect whether a news article is Fake or Real using Machine Learning & NLP.
This project builds a web-based application that classifies news text as Fake or Real, using a Logistic Regression model trained on a labeled dataset of ~45,000 articles. It is especially tuned for political and global news categories.

🚀 Live Demo
Glimpses:
<img width="905" height="623" alt="image" src="https://github.com/user-attachments/assets/5a95feee-a39b-4d51-a522-8484927013fd" />
<img width="930" height="748" alt="image" src="https://github.com/user-attachments/assets/29ce2fbf-3a90-4bd8-83dd-750ee893e75c" />

✅ Project Highlights
🧠 Model: Logistic Regression (98.4% Accuracy)

🧾 Vectorizer: TF-IDF with 5000 max features

🛠️ Frameworks: Streamlit (for UI), scikit-learn, NLTK

📊 Evaluation: Confusion Matrix, Precision, Recall, F1-Score

🧹 Text Preprocessing: Lowercasing, punctuation removal, stopwords filtering, stemming

🧠 Trained Categories:
politicsNews, worldnews, News, politics,
left-news, Government News, US_News, Middle-east

📈 Model Performance (Test Set)
Metric	Fake Class	Real Class
Precision	0.99	0.98
Recall	0.98	0.99
F1-Score	0.99	0.98
Accuracy	98.47%	–

🟩 Confusion Matrix

True Fake: 4604

True Real: 4239

False Fake: 45

False Real: 92

🖥️ Features in the App
✍️ Accepts any custom news input

🧹 Cleans & vectorizes using pre-trained TF-IDF

📊 Predicts with Logistic Regression and shows:

Result label (Fake or Real)

Confidence bar (color-coded)

Styled news preview card

📘 Explanation of model behavior

⚠️ Input limited to trained categories for accuracy

🧠 How It Works
Preprocessing:

Lowercase → Remove URLs, punctuation, stopwords → Optional stemming

Vectorization:

TF-IDF converts cleaned text into feature vectors

Prediction:

Logistic Regression outputs Fake or Real label

Probability score shown as confidence bar

🛑 Limitations
❌ Not trained on sports, entertainment, or finance news

⚠️ Accuracy may drop for unseen or out-of-domain categories

📜 License

Made with ❤️ by Shiv
