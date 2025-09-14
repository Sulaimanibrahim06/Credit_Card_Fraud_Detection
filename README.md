# 💳 Credit Card Fraud Detection Benchmark

This repository presents a machine learning project to detect fraudulent credit card transactions. It implements and evaluates multiple classification models—Logistic Regression, Random Forest, and XGBoost—to benchmark their performance on this important financial dataset.

Fraud detection is a critical task in the banking and financial sector, where fraudulent transactions must be identified accurately while minimizing false alarms. This project compares classical, ensemble, and boosting-based ML techniques, providing insights into model accuracy, recall, and robustness when handling highly imbalanced datasets.

## Models Implemented

-Logistic Regression: Establishes a simple linear baseline for fraud detection.

-Random Forest Classifier: An ensemble method that builds multiple decision trees to reduce overfitting and improve precision.

-XGBoost Classifier: A gradient boosting algorithm optimized for speed and performance on imbalanced datasets.

## Dataset

Source: [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?utm_source=chatgpt.com)
 (Kaggle)

Format: creditcard.csv file containing anonymized transaction features, amount, time, and class (0 = non-fraud, 1 = fraud).

Imbalance: Extremely imbalanced dataset (~0.17% fraud cases). To address this, SMOTE (Synthetic Minority Oversampling Technique) is applied.

## ⚖️ Evaluation Metrics

The following metrics were used to evaluate models on both training and testing sets:

Accuracy

Precision

Recall (Sensitivity)

F1-score

ROC-AUC

## 📈 Visualizations

Fraud vs Non-Fraud Distribution

Correlation Heatmap of Features

Model Performance Comparison (bar charts)

ROC Curves of all models

## 📊 Results & Discussion
| Model               | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| ------------------- | -------- | --------- | ------ | -------- | ------- |
| Logistic Regression | 0.977646     | 0.064293      | 0.878378 | 0.119816 | 0.967001   |
| Random Forest       | 0.999462  | 0.892308 | 0.783784 | 0.834532 | 0.962474    |
| XGBoost             | 0.999181  | 0.743750 | 0.804054 | 0.772727 | 0.973827    |


The Random Forest and XGBoost models achieved the best performance, significantly outperforming Logistic Regression, especially in recall and AUC. This indicates they are more effective in identifying fraudulent transactions. Logistic Regression, while interpretable, struggled with the imbalanced dataset.

## How to Use

Clone this repository:

git clone https://github.com/Sulaimanibrahim06/Credit_Card_Fraud_Detection.git
cd Credit_Card_Fraud_Detection


Install dependencies:

pip install -r requirements.txt


Open the Jupyter Notebook:

Run creditcard_fraud_detection.ipynb in Jupyter or Google Colab.

Execute all cells to load data, preprocess, train models, evaluate performance, and generate plots.

📌 Note: If GitHub fails to render the notebook properly, view it via [NBViewer](https://github.com/Sulaimanibrahim06/Credit_Card_Fraud_Detection/blob/main/CreditCardFruadDetection.ipynb)
.

## Dependencies

Python 3.x

pandas

numpy

scikit-learn

imbalanced-learn

xgboost

matplotlib

seaborn

Install with:

pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn

## Future Work

Deploying the model with Gradio or Streamlit for real-time fraud detection.

Trying Deep Learning models (Autoencoders, LSTMs).

Experimenting with real-time pipelines for live transaction monitoring.

## 📜 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 📬 Contact

For questions or feedback, feel free to open an issue or reach out at sulaimanibrahim3108@gmail.com.

