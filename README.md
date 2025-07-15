# stack-exchange-quetions
# 🧠 Auto Tag Prediction for Questions using Multi-Label Classification (NLP)
data set link-https://drive.google.com/drive/folders/1YA0pYS_kLgR-W_lFURzRs2Vd39U0hn-2?usp=sharing

This project builds a machine learning model to **automatically predict relevant tags** (like in Stack Overflow or Quora) based on the content of user-submitted questions. It applies **Natural Language Processing (NLP)** techniques and **multi-label classification** to assign one or more appropriate tags to each question.

---

## 📌 Problem Statement

Many Q&A platforms (e.g., Stack Overflow) require users to tag their questions properly for easy discovery and categorization. However, not all users are familiar with tagging or the right tags. This project solves that by training a machine learning model to **predict the most appropriate tags** for a question based on its text content.

---

## 📂 Dataset Overview

The dataset includes two CSV files:
- `Questions.csv`: Contains question IDs and their full HTML body.
- `Tags.csv`: Contains tag(s) assigned to each question.

Each question may have multiple associated tags, making this a **multi-label classification** problem.

---

## 🧼 Preprocessing Steps

- **HTML tag removal** using `BeautifulSoup`
- **Regex-based cleaning** to retain only alphabetic characters
- **Lowercasing**, **whitespace normalization**
- **Lemmatization and stopword removal** using `spaCy`
- Combined `Questions.csv` and `Tags.csv` on `Id`

---

## 🔠 Feature Engineering

- Cleaned questions were converted into numerical features using **TF-IDF Vectorization**
- Tags were binarized using **MultiLabelBinarizer**

---

## 🧠 Models Used

### 🔸 Multinomial Naive Bayes (Baseline)
- Applied on TF-IDF features
- Custom probability threshold selected using weighted F1 score

### 🔸 Logistic Regression with OneVsRestClassifier
- Handles multi-label cases effectively
- Chosen for better performance than Naive Bayes

---

## 🧪 Threshold Optimization

```python
def optimum_threshold(actual, pred_prob):
    thresholds = np.arange(0, 0.5, 0.01)
    scores = []
    for value in thresholds:
        pred_classes = classify(pred_prob, value)
        scores.append(f1_score(actual, pred_classes, average="weighted"))
    return thresholds[np.argmax(scores)]
