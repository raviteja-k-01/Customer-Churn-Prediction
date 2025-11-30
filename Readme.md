# Customer Churn Prediction

### Predicting whether a customer will stay or leave using machine learning (XGBoost)

---

## About the Project

This project is focused on predicting customer churn for an e-commerce platform.  
The idea was simple — given a customer’s activity, engagement, and satisfaction data,  
can we predict whether they’re likely to stop using the service?

It’s a fairly common business problem, but I wanted to build it **end-to-end** —  
from raw data to a working prediction script.  
Everything is done in Python with libraries like pandas, scikit-learn, and XGBoost.

---

## What I Did

- Cleaned and preprocessed messy data (lots of missing values and inconsistent text formats)
- Created new features that made more sense from a business perspective (like loyalty and engagement)
- Trained both Logistic Regression and XGBoost models to compare performance
- Saved the best model using pickle so it can be used later for predictions
- Wrote a simple Python script that takes a new customer’s info and tells if they’re likely to churn or not

I didn’t use Streamlit or Flask yet — wanted to keep it clean and focused on the model pipeline itself.

---

## Dataset Overview

The dataset has around **50,000 customers** and includes things like:
- Tenure (how long they’ve been a customer)
- App usage time
- Payment methods
- Complaints
- Order counts, cashback amounts, etc.
- Satisfaction score  
- And finally the **target variable: Churn (1 = churn, 0 = stay)**

The data is synthetic but structured to feel like something from a real-world e-commerce setup.

---

## Feature Engineering

Some of the new columns I created to make patterns more visible to the model:

- **LoyaltyScore** = Tenure × SatisfactionScore  
- **AvgSpendPerOrder** = TotalSpendLastYear / OrderCount  
- **EngagementScore** = (HourSpendOnApp × 0.7) + (EmailEngagementScore × 0.3)  
- **CashbackEfficiency** = CashbackAmount / TotalSpendLastYear  

These features actually helped make the relationships more realistic — for example, customers with high loyalty and low complaints usually stayed.

---

## Models and Results

I started with Logistic Regression as a baseline (because it’s easy to interpret).  
Then moved to **XGBoost**, which worked better for structured tabular data.

Parameters I used for XGBoost:
```python
XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=8,
    eval_metric='logloss',
    random_state=42
)
```

### Results (on a balanced dataset)
| Model | Accuracy | ROC-AUC | F1 |
|--------|-----------|---------|----|
| Logistic Regression | ~0.78 | 0.74 | 0.75 |
| XGBoost | **~0.90** | **0.87** | **0.88** |

These numbers will vary depending on the data version, but XGBoost consistently outperformed the logistic baseline.

---

## How to Use

1. Clone the repo  
   ```bash
   git clone https://github.com/yourusername/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. Install the dependencies  
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Run the notebook to retrain the model  
   ```bash
   jupyter notebook src/train_model.ipynb
   ```

4. Predict for a new customer  
   ```bash
   python src/predict_customer.py
   ```

You’ll see output like:
```
✅ Customer likely to STAY (Confidence: 0.84)
```
or
```
⚠️ Customer likely to CHURN (Confidence: 0.78)
```

---

## Project Folder Structure

```
CustomerChurnPrediction/
│
├── data/
│   └── E Commerce Dataset.xlsx
│
├── model/
│   └── churn_model.pkl
│
├── src/
│   ├── code.ipynb
│   ├── predict_customer.py
│
├── screenshots/
│   ├── churn_prediction_output.png
│   ├── feature_importance_plot.png
│   ├── confusion_matrix.png
│   └── accuracy_report.png
│
├── requirements.txt
├── README.md
└── .gitignore

```

---

## Tech Used

- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- Matplotlib / Seaborn  
- Pickle for model storage  

---

## What I Learned

- Handling messy categorical data is harder than it looks.
- Feature engineering is where most of the improvement came from — not just the model.
- It’s easy to overfit on synthetic data; regularization and evaluation metrics matter.
- Having a simple prediction script really helps show the project end-to-end.

---

## Next Steps

- Build a small web interface with Streamlit for user-friendly predictions  
- Try SHAP or LIME for model explainability  
- Possibly deploy the model on AWS or Render for demonstration  

---

## Author

**Ravi Teja Kesagani**  
Email: raviteja.inboxx@gmail.com
LinkedIn: [linkedin.com/in/ravitejakesagani1](https://www.linkedin.com/in/ravitejakesagani1/)  
GitHub: [github.com/raviteja-k-01](https://github.com/raviteja-k-01)

---

> “The best models are simple, explainable, and built with curiosity.”
