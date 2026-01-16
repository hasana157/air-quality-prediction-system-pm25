
# **PM2.5 Air Quality Prediction System**

A complete **end-to-end machine learning pipeline** for predicting Beijing’s hourly air pollution levels (PM2.5).
Includes **EDA, preprocessing, feature engineering, ML models, and a Streamlit dashboard**.

---

## 🚀 Project Overview

Air pollution is a major environmental threat affecting millions of people.
This project forecasts **PM2.5 concentration** using historical meteorological and air quality data from Beijing (2010–2014).

The system:

* Cleans raw data
* Handles missing values
* Extracts key features
* Trains ML/Nn models
* Evaluates performance
* Visualizes insights
* Provides a **GUI app** for interactive use

---

## 📂 Repository Structure

```
├── data/
│   └── beijing_aqi.csv
├── notebooks/
│   └── exploration_and_modeling.ipynb
├── src/
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── train_models.py
│   └── app.py
├── models/
│   ├── linear_regression.pkl
│   └── ann_model.h5
├── report/
│   └── Project Report.pdf
└── README.md
```

---

## 📊 Dataset

**Source:** UCI & Kaggle — Beijing PM2.5

* **Records:** 43,824 hourly samples
* **Target:** PM2.5 concentration (µg/m³)
* **Features:** datetime, temperature, dew point, pressure, wind speed/direction, precipitation

---

## 🧹 Data Preprocessing Highlights

✔ Merge Y/M/D/H into datetime
✔ Remove missing PM2.5 rows
✔ Fill missing meteorological values (forward/backward fill)
✔ One-hot encode wind direction (`cbwd`)
✔ Remove extreme outliers (>500 μg/m³)
✔ StandardScaler normalization
✔ Chronological train/test split (80/20)

---

## 🔍 Exploratory Data Analysis

Includes:

* PM2.5 distributions
* Yearly/monthly/hourly trends
* Seasonal peaks (winter)
* Correlation heatmaps
* Boxplots + outlier insights

💡 **Insight:** Winter dominates pollution due to heating + weather inversion.

*Add screenshots here:*
`/assets/eda_plot.png`

---

## ⚙️ Feature Engineering

* **Time features:** hour, weekday, month, season
* **Cyclical encoding:** sin/cos(hour)
* **Rolling windows:** 3/6/12/24h averages
* **Lag features:** previous PM2.5 values
* **Wind direction:** one-hot

These allow models to learn **temporal + weather patterns.**

---

## 🤖 Models Implemented

### 1️⃣ Linear Regression (Baseline)

* RMSE: **35–40**
* R²: **~0.55**
* Good for interpretability
* Misses non-linear atmospheric interactions

### 2️⃣ Artificial Neural Network (ANN)

* Layers: **Input → 32 → 16 → Output**
* Activation: **Sigmoid**
* Optimizer: **Gradient Descent**
* RMSE: **25–30**
* R²: **0.75–0.80**
* Captures seasonal + dynamic relationships

📌 **Winner:** ANN clearly outperforms LR.

---

## 📈 Model Comparison

| Model             | RMSE      | R²            |
| ----------------- | --------- | ------------- |
| Linear Regression | 35–40     | ~0.55         |
| ANN               | **25–30** | **0.75–0.80** |

*Add graph screenshot:*
`/assets/model_compare.png`

---

## 🎯 Predictions

Outputs:

* Actual vs Predicted curves
* Residual plots
* Error distributions

*Insert your graphs here.*

---

## 🖥 Streamlit GUI

Features:
✔ Data upload
✔ Cleaning + preprocessing viewer
✔ EDA dashboards
✔ Feature engineering workflow
✔ Model training + metrics
✔ Prediction visualization

Run App:

```bash
streamlit run app.py
```

*Add screenshots for each tab.*

---

## 💪 Strengths

* Handles noise + missing data
* Solid ANN forecasting accuracy
* Captures seasonality
* User-friendly interface

## ⚠️ Limitations

* Hourly only — no long-range forecast
* Weather anomalies lower accuracy
* No external events (fires, holidays, dust storms)

---

## 📌 Conclusion

This system proves that **ML + careful preprocessing + temporal intelligence** can reliably predict PM2.5.
ANN significantly enhances forecasting accuracy and enables a visual, interactive experience through Streamlit.

---

## 🗒 References

* UCI Air Quality Dataset
* Kaggle Beijing PM2.5
* Géron — *Hands-On ML with Scikit-Learn & TensorFlow*
* ISLR — *Introduction to Statistical Learning*
* McKinney — *Python for Data Analysis*

---

## 🙌 Contributors

**Hasana Zahid – CIIT/SP24-BAI-060**
**Dur-e-Shahwar – CIIT/SP24-BAI-013**
Instructor: **Dr. Usman Yaseen**

---

## ⭐ Future Improvements

* Add LSTM/GRU for sequence modeling
* Forecast days instead of hours
* Use external signals (traffic, weather APIs)
* Deploy on cloud (HuggingFace, Railway, Streamlit Cloud)

---

