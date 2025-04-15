import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    # chuyển đổi nhãn thành số
    df['category'] = df['category'].map({'ham': 0, 'junk': 1})
    return df

# trích xuất đặc trưng TF-IDF
def extract_features(df):
    vectorizer = TfidfVectorizer(
        max_features=5000,  # giới hạn số lượng từ
        stop_words='english',  # loại bỏ stop words
        ngram_range=(1, 2)  # xét cả unigram và bigram
    )
    X = vectorizer.fit_transform(df['message'])
    y = df['category']
    return X, y, vectorizer

def train_model(X_train, y_train):
    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver='lbfgs'
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # vẽ confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def main():
    df = load_data('../resource/email_data.csv')
    df = preprocess_data(df)

    X, y, vectorizer = extract_features(df)
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    import joblib
    joblib.dump(model, 'email_classifier.joblib')
    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

if __name__ == "__main__":
    main() 