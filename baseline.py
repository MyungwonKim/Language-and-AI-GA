import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# ==========================================
# CONFIGURATION
# ==========================================
FILE_PATH = "extrovert_introvert.csv"  # CHANGE THIS for each file
TARGET_COL = 'extrovert'          # CHANGE THIS (0/1 column)
USER_ID_COL = 'auhtor_ID'
TEXT_COL = 'post'
# ==========================================

def run_baselines():
    # 1. Load Data
    print(f"--- Loading {FILE_PATH} ---")
    df = pd.read_csv(FILE_PATH)
    df = df.dropna(subset=[TEXT_COL, TARGET_COL, USER_ID_COL])
    
    # 2. GROUP SPLIT (MUST match your Main Model split!)
    # We use user_id to ensure no user is in both train and test
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(df, groups=df[USER_ID_COL]))

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    X_train = train_df[TEXT_COL]
    y_train = train_df[TARGET_COL]
    X_test = test_df[TEXT_COL]
    y_test = test_df[TARGET_COL]

    print(f"Train Size: {len(X_train)} | Test Size: {len(X_test)}")
    print("-" * 30)

    # ==========================================
    # BASELINE 1: Majority Class (Zero-Rule)
    # ==========================================
    print("running Baseline 1: Majority Class...")
    
    # Strategy 'most_frequent' always predicts the class that appears most in y_train
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train, y_train)
    
    dummy_preds = dummy_clf.predict(X_test)
    
    print(">> Results for Majority Class:")
    print(f"Accuracy: {accuracy_score(y_test, dummy_preds):.4f}")
    # We use zero_division=0 to hide warnings if one class is never predicted
    print(classification_report(y_test, dummy_preds, zero_division=0)) 
    print("-" * 30)

    # ==========================================
    # BASELINE 2: TF-IDF + Logistic Regression
    # ==========================================
    print("running Baseline 2: TF-IDF + LogReg...")

    # We use a Pipeline to bundle the vectorizer and the model together
    pipeline = Pipeline([
        # 1. Turn text into numbers. 
        # stop_words='english' removes "the", "is", etc.
        # max_features=5000 keeps only the top 5,000 most common words to keep it fast
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
        
        # 2. The Classifier
        ('clf', LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)
    lr_preds = pipeline.predict(X_test)

    print(">> Results for TF-IDF + LogReg:")
    print(f"Accuracy: {accuracy_score(y_test, lr_preds):.4f}")
    print(classification_report(y_test, lr_preds))
    print("-" * 30)

if __name__ == "__main__":
    run_baselines()
    