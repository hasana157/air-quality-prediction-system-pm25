
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
<img width="967" height="690" alt="image" src="https://github.com/user-attachments/assets/775e0c64-032a-4de8-856c-12037d2d0608" />

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

<img width="708" height="293" alt="image" src="https://github.com/user-attachments/assets/09c249ca-2e5e-4e31-bc30-c4789b643426" />

`/assets/model_compare.png`

---

## 🎯 Predictions

Outputs:

* Actual vs Predicted curves
* Residual plots
* Error distributions

<img width="1730" height="562" alt="image" src="https://github.com/user-attachments/assets/a1aff8d0-2319-45b8-8c24-9401516853b8" />
<img width="847" height="501" alt="image" src="https://github.com/user-attachments/assets/99f632ac-db05-46f3-be13-a0de9154b0e5" />
<img width="844" height="487" alt="image" src="https://github.com/user-attachments/assets/1917e283-240e-49f1-96d0-0dcab07daa8e" />




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

<img width="328" height="441" alt="image" src="https://github.com/user-attachments/assets/bd9bbd2b-4a23-4593-8631-4f28d7c04037" />

<img width="882" height="549" alt="image" src="https://github.com/user-attachments/assets/9f67a3fd-218e-4cab-88e4-2e6723d0c33e" />

<img width="890" height="523" alt="image" src="https://github.com/user-attachments/assets/19b80811-91b0-410c-aaff-e67069987c8d" />

<img width="859" height="492" alt="image" src="https://github.com/user-attachments/assets/23bace91-2776-492b-9bbb-8b1eabe917a1" />

<img width="748" height="894" alt="image" src="https://github.com/user-attachments/assets/4e94d886-1ed8-44ef-a766-6e344069f2a9" />

<img width="630" height="716" alt="image" src="https://github.com/user-attachments/assets/6cc5e0b4-f97f-4327-a7c7-0832dcbba50d" />
<img width="1036" height="929" alt="image" src="https://github.com/user-attachments/assets/968f3156-a1e3-428e-856e-988e7ef0454f" />
<img width="635" height="350" alt="image" src="https://github.com/user-attachments/assets/b461a06d-959c-4128-b3bf-97148a8c2d25" />

<img width="435" height="279" alt="image" src="https://github.com/user-attachments/assets/e1872d8a-1dbf-49f6-9f63-75c46e055ba2" />
<img width="640" height="757" alt="image" src="https://github.com/user-attachments/assets/8d149d32-39d5-4083-b6d1-e61f7ed5c669" />
<img width="880" height="1039" alt="image" src="https://github.com/user-attachments/assets/37b04f94-291e-4876-adf3-ef6007ed2b84" />











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

