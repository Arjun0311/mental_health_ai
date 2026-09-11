"""
Trains and selects the best text-classification model for the emotion
classifier demo.

Improvements over the original version:
- Larger, balanced dataset (see generate_dataset.py) instead of 5 rows
- Stratified train/test split (keeps class balance in both sets)
- Compares three model families (Logistic Regression, Linear SVM, Naive Bayes)
  with 5-fold cross-validation instead of picking one blindly
- TF-IDF with unigrams+bigrams and sublinear TF scaling
- Wraps the chosen model in CalibratedClassifierCV when needed so the app
  can show a meaningful confidence score (predict_proba)
- Prints a full classification report + confusion matrix on the held-out
  test set, and saves a metrics.json alongside the model
- Saves the list of class labels so the app doesn't need to guess them
"""

import json
import pickle

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

RANDOM_STATE = 42

print("Loading dataset...")
data = pd.read_csv("dataset.csv")
data = data.dropna(subset=["text", "emotion"])
print(f"Loaded {len(data)} rows across {data['emotion'].nunique()} classes:")
print(data["emotion"].value_counts().to_string())

X_text = data["text"]
y = data["emotion"]

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.9,
    sublinear_tf=True,
)
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

candidates = {
    "logistic_regression": LogisticRegression(max_iter=1000, C=3.0),
    "linear_svm": LinearSVC(C=1.0),
    "naive_bayes": MultinomialNB(),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

print("\nComparing candidate models with 5-fold cross-validation on the training set...")
cv_results = {}
for name, clf in candidates.items():
    scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy")
    cv_results[name] = scores.mean()
    print(f"  {name:22s} mean CV accuracy = {scores.mean():.3f} (+/- {scores.std():.3f})")

best_name = max(cv_results, key=cv_results.get)
print(f"\nSelected best model: {best_name}")

best_clf = candidates[best_name]

# LinearSVC has no predict_proba; calibrate it so the app can show confidence.
if best_name == "linear_svm":
    final_model = CalibratedClassifierCV(best_clf, cv=cv)
else:
    final_model = best_clf

final_model.fit(X_train, y_train)

# Evaluate on the held-out test set
y_pred = final_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
labels_sorted = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

print(f"\nHeld-out test accuracy: {test_accuracy:.3f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred))
print("Confusion matrix (rows=true, cols=predicted):")
print(f"  labels: {labels_sorted}")
print(cm)

# Save model, vectorizer, and metadata
pickle.dump(final_model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

metrics = {
    "model_selected": best_name,
    "cv_accuracy_by_model": cv_results,
    "test_accuracy": test_accuracy,
    "classification_report": report,
    "confusion_matrix": cm.tolist(),
    "labels": labels_sorted,
    "n_train": int(len(y_train)),
    "n_test": int(len(y_test)),
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved model.pkl, vectorizer.pkl, and metrics.json")
