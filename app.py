"""
Beijing PM2.5 Air Quality Prediction System - Streamlit Application
====================================================================
Complete interactive ML pipeline for air quality prediction

Authors: Hasana Zahid (SP24-BAI-060) & Dur-e-Shahwar (SP24-BAI-013)
Institution: COMSATS University Islamabad

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Beijing PM2.5 Prediction System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        color: #FFFFFF;
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.4);
        animation: gradient 3s ease infinite;
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.5);
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 0.75rem 2.5rem;
        font-weight: bold;
        font-size: 1.1rem;
        border: none;
        transition: all 0.3s;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(17, 153, 142, 0.3);
        font-weight: bold;
    }
    .info-box {
        background: linear-gradient(135deg, #3a7bd5 0%, #00d2ff 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(58, 123, 213, 0.3);
    }
    .warning-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(240, 147, 251, 0.3);
    }
    .feature-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        font-weight: bold;
        box-shadow: 0 4px 12px rgba(250, 112, 154, 0.3);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    h1, h2, h3 {
        color: #667eea;
        font-weight: bold;
    }
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# NEURAL NETWORK CLASS
# =============================================================================
class NeuralNetwork:
    """Feedforward neural network with backpropagation"""
    
    def __init__(self, input_size, hidden_sizes=[64, 32, 16], learning_rate=0.001):
        self.lr = learning_rate
        self.layers = []
        
        layer_sizes = [input_size] + hidden_sizes + [1]
        for i in range(len(layer_sizes) - 1):
            # He initialization for better training
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.layers.append({'w': w, 'b': b})
    
    def sigmoid(self, x):
        # Clip values to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        
        for i, layer in enumerate(self.layers):
            z = np.dot(self.activations[-1], layer['w']) + layer['b']
            self.z_values.append(z)
            
            # Use sigmoid for hidden layers and linear for output
            if i < len(self.layers) - 1:
                a = self.sigmoid(z)
            else:
                a = z  # Linear activation for regression output
            self.activations.append(a)
        
        return self.activations[-1]
    
    def backward(self, X, y):
        m = X.shape[0]
        dz = self.activations[-1] - y.reshape(-1, 1)
        
        for i in range(len(self.layers) - 1, -1, -1):
            dw = np.dot(self.activations[i].T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            
            # Gradient clipping for stability
            dw = np.clip(dw, -5.0, 5.0)
            db = np.clip(db, -5.0, 5.0)
            
            # Gradient descent update
            self.layers[i]['w'] -= self.lr * dw
            self.layers[i]['b'] -= self.lr * db
            
            if i > 0:
                dz = np.dot(dz, self.layers[i]['w'].T) * self.sigmoid_derivative(self.z_values[i-1])
    
    def train(self, X_train, y_train, X_test, y_test, epochs=200, progress_bar=None, status_text=None):
        train_losses = []
        test_losses = []
        
        for epoch in range(epochs):
            y_pred_train = self.forward(X_train)
            train_loss = np.mean((y_pred_train.flatten() - y_train) ** 2)
            
            if np.isnan(train_loss):
                st.error(f"⚠️ NaN detected at epoch {epoch+1}!")
                break
            
            train_losses.append(train_loss)
            self.backward(X_train, y_train)
            
            y_pred_test = self.forward(X_test)
            test_loss = np.mean((y_pred_test.flatten() - y_test) ** 2)
            test_losses.append(test_loss)
            
            if progress_bar and (epoch + 1) % 10 == 0:
                progress_bar.progress((epoch + 1) / epochs)
                if status_text:
                    status_text.text(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        
        return train_losses, test_losses
    
    def predict(self, X):
        return self.forward(X).flatten()

# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================

@st.cache_data
def load_data(uploaded_file):
    """Load Beijing PM2.5 dataset"""
    df = pd.read_csv(uploaded_file)
    return df

def preprocess_data(df):
    """Clean and prepare data"""
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    
    initial_rows = len(df)
    df = df.dropna(subset=['pm2.5']).copy()
    dropped = initial_rows - len(df)
    
    weather_cols = ['DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']
    for col in weather_cols:
        if col in df.columns:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
    
    if 'cbwd' in df.columns:
        df['cbwd'] = df['cbwd'].fillna('Unknown')
    
    df = df[df['pm2.5'] < 500].copy()
    df = df.sort_values('datetime').reset_index(drop=True)
    
    return df, dropped

def create_features(df):
    """Engineer features"""
    df = df.copy()
    
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['season'] = df['month'].apply(lambda x: (x%12 + 3)//3)
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    for window in [3, 6, 12, 24]:
        df[f'pm2.5_rolling_{window}h'] = df['pm2.5'].rolling(window=window, min_periods=1).mean()
    
    for lag in [1, 3, 6, 12, 24]:
        df[f'pm2.5_lag_{lag}h'] = df['pm2.5'].shift(lag)
    
    if 'cbwd' in df.columns:
        wind_dummies = pd.get_dummies(df['cbwd'], prefix='wind')
        df = pd.concat([df, wind_dummies], axis=1)
    
    df = df.dropna().reset_index(drop=True)
    return df

def prepare_train_test(df, test_size=0.2):
    """Split data"""
    feature_cols = ['hour', 'day', 'month', 'dayofweek', 'season', 
                   'hour_sin', 'hour_cos', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir',
                   'pm2.5_rolling_3h', 'pm2.5_rolling_6h', 'pm2.5_rolling_12h', 'pm2.5_rolling_24h',
                   'pm2.5_lag_1h', 'pm2.5_lag_3h', 'pm2.5_lag_6h', 'pm2.5_lag_12h', 'pm2.5_lag_24h']
    
    wind_cols = [col for col in df.columns if col.startswith('wind_')]
    feature_cols.extend(wind_cols)
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols].values
    y = df['pm2.5'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=False
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler, feature_cols

def evaluate_model(y_true, y_pred, model_name):
    """Evaluate model"""
    return {
        'Model': model_name,
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R² Score': r2_score(y_true, y_pred)
    }

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'preprocessed' not in st.session_state:
    st.session_state.preprocessed = False
if 'features_engineered' not in st.session_state:
    st.session_state.features_engineered = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

# =============================================================================
# MAIN APP
# =============================================================================

st.markdown('<h1 class="main-header">🌍 Beijing PM2.5 Air Quality Prediction System</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 Navigation")
    page = st.radio("", 
                    ["🏠 Home",
                     "📊 Data Upload", 
                     "🔧 Preprocessing", 
                     "📈 EDA", 
                     "⚙️ Feature Engineering",
                     "🤖 Model Training",
                     "📉 Model Comparison",
                     "🎯 Predictions"],
                    label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("### 📚 Project Info")
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1.5rem; border-radius: 15px; color: white; 
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);'>
        <p style='font-size: 1.1rem; margin: 0;'><strong>🏛️ COMSATS University Islamabad</strong></p>
        <br>
        <p style='margin: 0.3rem 0;'><strong>👩‍💻 Developers:</strong></p>
        <p style='margin: 0.2rem 0; padding-left: 1rem;'>• Hasana Zahid<br>&nbsp;&nbsp;(SP24-BAI-060)</p>
        <p style='margin: 0.2rem 0; padding-left: 1rem;'>• Dur-e-Shahwar<br>&nbsp;&nbsp;(SP24-BAI-013)</p>
        <br>
        <p style='margin: 0.3rem 0;'><strong>👨‍🏫 Instructors:</strong></p>
        <p style='margin: 0.2rem 0; padding-left: 1rem;'>• Dr.Usman Yaseen</p>
        
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📊 Progress")
    if st.session_state.data_loaded:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                    padding: 0.8rem; border-radius: 10px; color: white; 
                    margin: 0.5rem 0; font-weight: bold; text-align: center;'>
            ✅ Data Loaded
        </div>
        """, unsafe_allow_html=True)
    if st.session_state.preprocessed:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                    padding: 0.8rem; border-radius: 10px; color: white; 
                    margin: 0.5rem 0; font-weight: bold; text-align: center;'>
            ✅ Preprocessed
        </div>
        """, unsafe_allow_html=True)
    if st.session_state.features_engineered:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                    padding: 0.8rem; border-radius: 10px; color: white; 
                    margin: 0.5rem 0; font-weight: bold; text-align: center;'>
            ✅ Features Ready
        </div>
        """, unsafe_allow_html=True)
    if st.session_state.models_trained:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                    padding: 0.8rem; border-radius: 10px; color: white; 
                    margin: 0.5rem 0; font-weight: bold; text-align: center;'>
            ✅ Models Trained
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# PAGE 0: HOME
# =============================================================================
if page == "🏠 Home":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 🌍 Welcome to Beijing PM2.5 Prediction System")
        
        st.markdown("""
        ### 📖 About This Project
        
        This comprehensive machine learning system predicts Beijing's PM2.5 air quality levels using 
        historical meteorological and pollution data from **2010-2014**. The project implements a 
        complete ML pipeline with custom neural network implementation.
        
        ### 🎯 Key Features
        
        - **Advanced Preprocessing**: Time-based feature engineering and rolling statistics
        - **Exploratory Analysis**: Comprehensive visualization of pollution patterns
        - **Feature Engineering**: 26 engineered features including cyclical encoding and lag features
        - **Multiple Models**: Linear Regression baseline + Custom Neural Network with backpropagation
        - **Gradient Descent**: Manual implementation with gradient clipping
        - **Interactive Predictions**: Real-time PM2.5 forecasting
        
        ### 📊 Dataset Overview
        
        - **Records**: ~41,600 hourly observations
        - **Time Period**: 2010-2014 (5 years)
        - **Location**: Beijing, China
        - **Features**: Temperature, pressure, wind, precipitation, and more
        
        ### 🚀 How to Use
        
        1. **Upload Data**: Start with Beijing PM2.5 CSV file
        2. **Preprocess**: Clean and transform the data
        3. **Explore**: Visualize temporal and seasonal patterns
        4. **Engineer Features**: Create 26 meaningful features
        5. **Train Models**: Compare Linear Regression vs Neural Network
        6. **Predict**: Make predictions on custom inputs
        
        ### 📈 Expected Performance
        
        - **Linear Regression**: R² ≈ 0.98, RMSE ≈ 11 μg/m³
        - **Neural Network**: R² ≈ 0.98, RMSE ≈ 12 μg/m³
        - **Features**: Lag features are most predictive
        """)
    
    with col2:
        st.markdown("### 🎓 Academic Context")
        st.markdown("""
        <div style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                    padding: 1.5rem; border-radius: 15px; color: white; 
                    box-shadow: 0 4px 12px rgba(250, 112, 154, 0.3);'>
            <p style='font-weight: bold; font-size: 1.2rem; margin: 0;'>📚 BS Artificial Intelligence</p>
            <br>
            <p style='margin: 0.3rem 0;'><strong>🏛️ COMSATS University Islamabad</strong></p>
            <br>
            <p style='margin: 0.3rem 0;'><strong>📅 Year:</strong> 2024-2028</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📚 Technologies Used")
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #3a7bd5 0%, #00d2ff 100%); 
                    padding: 1.5rem; border-radius: 15px; color: white; 
                    box-shadow: 0 4px 12px rgba(58, 123, 213, 0.3);'>
        """, unsafe_allow_html=True)
        
        technologies = {
            "Python": "🐍",
            "NumPy": "🔢",
            "Pandas": "🐼",
            "Scikit-learn": "🤖",
            "Matplotlib": "📊",
            "Plotly": "📉",
            "Streamlit": "⚡"
        }
        
        for tech, emoji in technologies.items():
            st.markdown(f"<p style='margin: 0.5rem 0; font-weight: bold;'>{emoji} {tech}</p>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🎯 Model Highlights")
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 15px; color: white; 
                    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);'>
            <p style='font-weight: bold; margin-bottom: 0.5rem;'>🧠 Neural Network Architecture:</p>
            <p style='margin: 0.3rem 0; padding-left: 1rem;'>• Input: 26 features</p>
            <p style='margin: 0.3rem 0; padding-left: 1rem;'>• Hidden: 64 → 32 → 16 neurons</p>
            <p style='margin: 0.3rem 0; padding-left: 1rem;'>• Output: 1 (PM2.5)</p>
            <p style='margin: 0.3rem 0; padding-left: 1rem;'>• Activation: Sigmoid</p>
            <p style='margin: 0.3rem 0; padding-left: 1rem;'>• Optimization: Gradient Descent</p>
            <p style='margin: 0.3rem 0; padding-left: 1rem;'>• Learning Rate: 0.001</p>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# PAGE 1: DATA UPLOAD
# =============================================================================
elif page == "📊 Data Upload":
    st.header("📊 Data Upload & Preview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Upload Your Dataset")
        st.info("📁 Upload the Beijing PM2.5 CSV file (PRSA_data_2010.1.1-2014.12.31.csv)")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df_original = df
                st.session_state.data_loaded = True
                
                st.markdown('<div class="success-box">✅ Dataset loaded successfully!</div>', unsafe_allow_html=True)
                
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("📊 Total Records", f"{df.shape[0]:,}")
                col_b.metric("📋 Features", df.shape[1])
                col_c.metric("📅 Years", f"{df['year'].min()}-{df['year'].max()}")
                col_d.metric("❓ Missing PM2.5", df['pm2.5'].isna().sum())
                
                st.markdown("### 📋 Dataset Preview (First 10 Rows)")
                st.dataframe(df.head(10), use_container_width=True)
                
                st.markdown("### 📊 Column Information")
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes,
                    'Non-Null': df.count(),
                    'Unique': [df[col].nunique() for col in df.columns]
                })
                st.dataframe(col_info, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
    
    with col2:
        if st.session_state.data_loaded:
            st.markdown("### 📈 Quick Statistics")
            df = st.session_state.df_original
            
            st.markdown("#### Missing Values")
            missing = df.isnull().sum()
            if missing.sum() == 0:
                st.success("✅ No missing values!")
            else:
                st.warning(f"⚠️ Found {missing.sum()} missing values")
                st.dataframe(missing[missing > 0])
            
            if 'pm2.5' in df.columns:
                pm25_data = df['pm2.5'].dropna()
                st.markdown("#### PM2.5 Statistics")
                st.metric("Mean", f"{pm25_data.mean():.2f} μg/m³")
                st.metric("Median", f"{pm25_data.median():.2f} μg/m³")
                st.metric("Max", f"{pm25_data.max():.2f} μg/m³")
            
            st.markdown("#### 💾 Download Sample")
            csv = df.head(100).to_csv(index=False)
            st.download_button(
                "Download Sample CSV",
                csv,
                "beijing_pm25_sample.csv",
                "text/csv"
            )

# =============================================================================
# PAGE 2: PREPROCESSING
# =============================================================================
elif page == "🔧 Preprocessing":
    st.header("🔧 Data Preprocessing")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first!")
    else:
        st.markdown("### Preprocessing Pipeline")
        
        with st.expander("📖 View Preprocessing Steps", expanded=True):
            st.markdown("""
            #### Operations:
            
            1. **Datetime Creation**
               - Combine year, month, day, hour into datetime column
            
            2. **Missing Value Treatment**
               - Drop rows with missing PM2.5 (target variable)
               - Forward fill weather data
            
            3. **Outlier Removal**
               - Remove PM2.5 > 500 μg/m³ (measurement errors)
            
            4. **Sorting**
               - Sort by datetime for time series analysis
            """)
        
        if st.button("🚀 Run Preprocessing", type="primary", use_container_width=True):
            with st.spinner("⏳ Processing data..."):
                df_processed, dropped_rows = preprocess_data(st.session_state.df_original)
                st.session_state.df_processed = df_processed
                st.session_state.preprocessed = True
            
            st.balloons()
            st.markdown('<div class="success-box">✅ Preprocessing completed!</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Before")
                st.metric("Records", f"{st.session_state.df_original.shape[0]:,}")
                st.metric("Columns", st.session_state.df_original.shape[1])
                st.dataframe(st.session_state.df_original.head(), use_container_width=True)
            
            with col2:
                st.markdown("### ✨ After")
                st.metric("Records", f"{df_processed.shape[0]:,}")
                st.metric("Columns", df_processed.shape[1])
                st.metric("Dropped", f"{dropped_rows:,}")
                st.dataframe(df_processed.head(), use_container_width=True)
            
            st.markdown("### 📊 PM2.5 Distribution")
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            data = df_processed['pm2.5']
            axes[0].hist(data, bins=50, color='#1976D2', edgecolor='black', alpha=0.7)
            axes[0].set_title(f'PM2.5 Distribution\n(Skewness: {stats.skew(data):.2f})', fontweight='bold')
            axes[0].set_xlabel('PM2.5 (μg/m³)')
            axes[0].set_ylabel('Frequency')
            axes[0].grid(axis='y', alpha=0.3)
            
            axes[1].boxplot(data)
            axes[1].set_title('PM2.5 Box Plot', fontweight='bold')
            axes[1].set_ylabel('PM2.5 (μg/m³)')
            axes[1].grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)

# =============================================================================
# PAGE 3: EDA
# =============================================================================
elif page == "📈 EDA":
    st.header("📈 Exploratory Data Analysis")
    
    if not st.session_state.preprocessed:
        st.warning("⚠️ Please complete preprocessing first!")
    else:
        df = st.session_state.df_processed
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Temporal", "🌡️ Weather", "🔗 Correlations"])
        
        with tab1:
            st.subheader("📊 Data Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📝 Records", f"{df.shape[0]:,}")
            col2.metric("📅 Years", f"{df['year'].nunique()}")
            col3.metric("📊 Avg PM2.5", f"{df['pm2.5'].mean():.2f}")
            col4.metric("📈 Max PM2.5", f"{df['pm2.5'].max():.2f}")
            
            st.markdown("---")
            
            fig = make_subplots(rows=2, cols=2,
                subplot_titles=('PM2.5 Distribution', 'PM2.5 Over Time', 'Monthly Average', 'Hourly Average'))
            
            fig.add_trace(go.Histogram(x=df['pm2.5'], nbinsx=50), row=1, col=1)
            
            sample = df.sample(min(1000, len(df)))
            fig.add_trace(go.Scatter(x=sample['datetime'], y=sample['pm2.5'], mode='lines'), row=1, col=2)
            
            monthly = df.groupby('month')['pm2.5'].mean()
            fig.add_trace(go.Bar(x=monthly.index, y=monthly.values), row=2, col=1)
            
            hourly = df.groupby('hour')['pm2.5'].mean()
            fig.add_trace(go.Bar(x=hourly.index, y=hourly.values), row=2, col=2)
            
            fig.update_layout(height=700, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("📈 Temporal Trends")
            
            yearly = df.groupby('year')['pm2.5'].mean().reset_index()
            fig = px.line(yearly, x='year', y='pm2.5',
                        title='Average PM2.5 by Year',
                        markers=True)
            fig.update_traces(line_width=3)
            st.plotly_chart(fig, use_container_width=True)
            
            monthly = df.groupby('month')['pm2.5'].mean().reset_index()
            fig = px.bar(monthly, x='month', y='pm2.5',
                        title='Average PM2.5 by Month',
                        color='pm2.5')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("🌡️ Weather Impact")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.scatter(df.sample(1000), x='TEMP', y='pm2.5',
                               title='Temperature vs PM2.5',
                               trendline='ols')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(df.sample(1000), x='DEWP', y='pm2.5',
                               title='Dew Point vs PM2.5',
                               trendline='ols')
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("🔗 Correlations")
            
            corr_cols = ['pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']
            corr_data = df[corr_cols].corr()
            
            fig = px.imshow(corr_data, 
                          text_auto='.2f',
                          aspect='auto',
                          color_continuous_scale='RdBu_r')
            fig.update_layout(title='Feature Correlation Matrix')
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PAGE 4: FEATURE ENGINEERING
# =============================================================================
elif page == "⚙️ Feature Engineering":
    st.header("⚙️ Feature Engineering")
    
    if not st.session_state.preprocessed:
        st.warning("⚠️ Please complete preprocessing first!")
    else:
        with st.expander("📖 View Feature Engineering Steps", expanded=True):
            st.markdown("""
            #### Features Created:
            
            1. **Time Features** (7 features)
               - hour, day, month, dayofweek, season
               - hour_sin, hour_cos (cyclical encoding)
            
            2. **Rolling Averages** (4 features)
               - 3h, 6h, 12h, 24h rolling mean of PM2.5
            
            3. **Lag Features** (5 features)
               - PM2.5 from 1h, 3h, 6h, 12h, 24h ago
            
            4. **Meteorological** (6 features)
               - DEWP, TEMP, PRES, Iws, Is, Ir
            
            5. **Wind Direction** (one-hot encoded)
            
            **Total: 26+ features**
            """)
        
        if st.button("🔨 Generate Features", type="primary", use_container_width=True):
            with st.spinner("⏳ Engineering features..."):
                df_features = create_features(st.session_state.df_processed)
                st.session_state.df_features = df_features
                st.session_state.features_engineered = True
            
            st.balloons()
            st.markdown('<div class="success-box">✅ Features created!</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("📊 Total Features", len(df_features.columns))
            col2.metric("📝 Records", f"{df_features.shape[0]:,}")
            col3.metric("🎯 Target", "PM2.5")
            
            st.markdown("### 📊 Feature Preview")
            st.dataframe(df_features.head(10), use_container_width=True)
            
            st.markdown("### 📈 Feature Statistics")
            st.dataframe(df_features.describe(), use_container_width=True)

# =============================================================================
# PAGE 5: MODEL TRAINING
# =============================================================================
elif page == "🤖 Model Training":
    st.header("🤖 Model Training")
    
    if not st.session_state.features_engineered:
        st.warning("⚠️ Please complete feature engineering first!")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### Select Models")
            train_lr = st.checkbox("📊 Linear Regression (Baseline)", value=True)
            train_nn = st.checkbox("🧠 Neural Network (Backpropagation)", value=True)
        
        with col2:
            st.markdown("#### Training Parameters")
            test_size = st.slider("Test Set Size (%)", 10, 40, 20, 5)
            epochs = st.slider("NN Epochs", 50, 500, 200, 50)
            learning_rate = st.select_slider("Learning Rate",
                options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)
        
        if st.button("🚀 Train Models", type="primary", use_container_width=True):
            df_features = st.session_state.df_features
            
            with st.spinner("📊 Splitting data..."):
                X_train, X_test, y_train, y_test, scaler, feature_cols = prepare_train_test(
                    df_features, test_size=test_size/100)
                
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.scaler = scaler
                st.session_state.feature_cols = feature_cols
            
            st.success(f"✅ Data split: {len(X_train)} train, {len(X_test)} test")
            
            results = {}
            
            # Linear Regression
            if train_lr:
                st.markdown("### 📈 Training Linear Regression")
                with st.spinner("Training..."):
                    lr = LinearRegression()
                    lr.fit(X_train, y_train)
                    y_pred_lr = lr.predict(X_test)
                
                results['Linear Regression'] = {
                    'model': lr,
                    'predictions': y_pred_lr,
                    'metrics': evaluate_model(y_test, y_pred_lr, 'Linear Regression')
                }
                
                st.success("✅ Linear Regression trained!")
                col1, col2, col3 = st.columns(3)
                col1.metric("RMSE", f"{results['Linear Regression']['metrics']['RMSE']:.2f}")
                col2.metric("MAE", f"{results['Linear Regression']['metrics']['MAE']:.2f}")
                col3.metric("R²", f"{results['Linear Regression']['metrics']['R² Score']:.4f}")
            
            # Neural Network
            if train_nn:
                st.markdown("### 🧠 Training Neural Network")
                st.info(f"Architecture: {X_train.shape[1]} → 64 → 32 → 16 → 1 | Activation: Sigmoid | LR: {learning_rate}")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                nn = NeuralNetwork(X_train.shape[1], hidden_sizes=[64, 32, 16], learning_rate=learning_rate)
                
                train_losses, test_losses = nn.train(
                    X_train, y_train, X_test, y_test,
                    epochs=epochs, progress_bar=progress_bar, status_text=status_text)
                
                y_pred_nn = nn.predict(X_test)
                
                results['Neural Network'] = {
                    'model': nn,
                    'predictions': y_pred_nn,
                    'metrics': evaluate_model(y_test, y_pred_nn, 'Neural Network'),
                    'train_losses': train_losses,
                    'test_losses': test_losses
                }
                
                st.success("✅ Neural Network trained!")
                col1, col2, col3 = st.columns(3)
                col1.metric("RMSE", f"{results['Neural Network']['metrics']['RMSE']:.2f}")
                col2.metric("MAE", f"{results['Neural Network']['metrics']['MAE']:.2f}")
                col3.metric("R²", f"{results['Neural Network']['metrics']['R² Score']:.4f}")
                
                st.markdown("### 📉 Training Loss Curve")
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=train_losses, mode='lines', name='Train Loss'))
                fig.add_trace(go.Scatter(y=test_losses, mode='lines', name='Test Loss'))
                fig.update_layout(title='Gradient Descent Convergence',
                                xaxis_title='Epoch', yaxis_title='MSE Loss')
                st.plotly_chart(fig, use_container_width=True)
            
            st.session_state.results = results
            st.session_state.models_trained = True
            st.balloons()
            
            st.markdown("---")
            st.markdown("### 📊 Training Summary")
            results_df = pd.DataFrame([r['metrics'] for r in results.values()])
            
            def highlight_best(s):
                if s.name == 'R² Score':
                    is_max = s == s.max()
                    return ['background-color: #C8E6C9; font-weight: bold' if v else '' for v in is_max]
                elif s.name in ['RMSE', 'MAE']:
                    is_min = s == s.min()
                    return ['background-color: #C8E6C9; font-weight: bold' if v else '' for v in is_min]
                return ['']*len(s)
            
            styled_df = results_df.style.apply(highlight_best)
            st.dataframe(styled_df, use_container_width=True)

# =============================================================================
# PAGE 6: MODEL COMPARISON
# =============================================================================
elif page == "📉 Model Comparison":
    st.header("📉 Model Comparison & Evaluation")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first!")
    else:
        results = st.session_state.results
        
        st.markdown("### 📊 Model Performance Comparison")
        results_df = pd.DataFrame([r['metrics'] for r in results.values()])
        results_df = results_df.sort_values('R² Score', ascending=False)
        
        def highlight_best(s):
            if s.name == 'R² Score':
                is_max = s == s.max()
                return ['background-color: #C8E6C9; color: #1B5E20; font-weight: bold' if v else '' for v in is_max]
            elif s.name in ['RMSE', 'MAE']:
                is_min = s == s.min()
                return ['background-color: #C8E6C9; color: #1B5E20; font-weight: bold' if v else '' for v in is_min]
            return ['']*len(s)
        
        styled_df = results_df.style.apply(highlight_best).format({
            'RMSE': '{:.4f}',
            'MAE': '{:.4f}',
            'R² Score': '{:.4f}'
        })
        
        st.dataframe(styled_df, use_container_width=True)
        
        best_model = results_df.iloc[0]['Model']
        best_r2 = results_df.iloc[0]['R² Score']
        st.success(f"🏆 **Best Model**: {best_model} with R² = {best_r2:.4f}")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 R² Comparison", "📈 Predictions", "📉 Residuals", "🧠 NN Training"])
        
        with tab1:
            st.subheader("R² Score Comparison")
            
            fig = px.bar(results_df, x='Model', y='R² Score',
                        title='Model Performance (R² Score)',
                        color='R² Score',
                        color_continuous_scale='Blues',
                        text='R² Score')
            fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            fig.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Predicted vs Actual")
            
            model_select = st.selectbox("Select Model:", list(results.keys()))
            
            y_test = st.session_state.y_test
            y_pred = results[model_select]['predictions']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_test, y=y_pred,
                mode='markers',
                marker=dict(color='#1976D2', size=4, opacity=0.5),
                name='Predictions'
            ))
            
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Perfect'
            ))
            
            fig.update_layout(
                title=f'Actual vs Predicted PM2.5 - {model_select}',
                xaxis_title='Actual PM2.5 (μg/m³)',
                yaxis_title='Predicted PM2.5 (μg/m³)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            metrics = results[model_select]['metrics']
            col1, col2, col3 = st.columns(3)
            col1.metric("RMSE", f"{metrics['RMSE']:.2f}")
            col2.metric("MAE", f"{metrics['MAE']:.2f}")
            col3.metric("R²", f"{metrics['R² Score']:.4f}")
        
        with tab3:
            st.subheader("Residual Analysis")
            
            model_select = st.selectbox("Select Model:", list(results.keys()), key='res')
            
            y_test = st.session_state.y_test
            y_pred = results[model_select]['predictions']
            residuals = y_test - y_pred
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(residuals, nbins=50,
                                 title='Residual Distribution',
                                 labels={'value': 'Residuals'},
                                 color_discrete_sequence=['#1976D2'])
                fig.add_vline(x=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=y_pred, y=residuals,
                    mode='markers',
                    marker=dict(color='#1976D2', size=4, opacity=0.5)
                ))
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                fig.update_layout(
                    title='Residual Plot',
                    xaxis_title='Predicted PM2.5',
                    yaxis_title='Residuals'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("🧠 Neural Network Training")
            
            if 'Neural Network' in results and 'train_losses' in results['Neural Network']:
                train_losses = results['Neural Network']['train_losses']
                test_losses = results['Neural Network']['test_losses']
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=train_losses, mode='lines', 
                                        name='Train Loss', line=dict(width=2)))
                fig.add_trace(go.Scatter(y=test_losses, mode='lines',
                                        name='Test Loss', line=dict(width=2)))
                fig.update_layout(
                    title='Training Progress (Gradient Descent)',
                    xaxis_title='Epoch',
                    yaxis_title='MSE Loss',
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2 = st.columns(2)
                col1.metric("Final Train Loss", f"{train_losses[-1]:.4f}")
                col2.metric("Final Test Loss", f"{test_losses[-1]:.4f}")
            else:
                st.info("Train Neural Network to see training history")

# =============================================================================
# PAGE 7: PREDICTIONS
# =============================================================================
elif page == "🎯 Predictions":
    st.header("🎯 Make Predictions")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first!")
    else:
        st.markdown("### Single Prediction")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 📝 Enter Information")
            
            year = st.number_input("Year", min_value=2010, max_value=2030, value=2024)
            month = st.slider("Month", 1, 12, 6)
            day = st.slider("Day", 1, 31, 15)
            hour = st.slider("Hour", 0, 23, 12)
            
            st.markdown("#### 🌡️ Weather Conditions")
            temp = st.slider("Temperature (°C)", -20, 40, 15)
            dewp = st.slider("Dew Point (°C)", -30, 30, 5)
            pres = st.slider("Pressure (hPa)", 990, 1050, 1015)
            
            model_choice = st.selectbox("Model", list(st.session_state.results.keys()))
        
        with col2:
            st.markdown("#### ℹ️ Prediction Info")
            st.info(f"""
            **Selected Inputs:**
            - Date: {year}-{month:02d}-{day:02d}
            - Time: {hour:02d}:00
            - Temp: {temp}°C
            - Dew Point: {dewp}°C
            - Pressure: {pres} hPa
            - Model: {model_choice}
            """)
        
        if st.button("🔮 Predict PM2.5", type="primary", use_container_width=True):
            # Note: This is simplified - in real use you'd need all 26 features
            st.warning("⚠️ For actual predictions, all 26 features (including lag and rolling features) are needed. This is a demo showing the interface.")
            
            # Demo prediction
            sample_pred = np.random.uniform(50, 150)
            
            # Air quality indicators
            st.markdown("""
            <div style='text-align: center; padding: 1rem;'>
                <div style='font-size: 3rem;'>
                    🔮 🌡️ 🌫️ 📊 ✨
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### 📊 Prediction Result")
            
            # Determine air quality level
            if sample_pred < 50:
                quality = "Good 😊"
                quality_color = "#00b09b"
                quality_icon = "🟢"
                advice = "Perfect day for outdoor activities!"
            elif sample_pred < 100:
                quality = "Moderate 😐"
                quality_color = "#f7b733"
                quality_icon = "🟡"
                advice = "Acceptable air quality for most people."
            elif sample_pred < 150:
                quality = "Unhealthy for Sensitive Groups ⚠️"
                quality_color = "#fc4a1a"
                quality_icon = "🟠"
                advice = "Sensitive individuals should limit outdoor exposure."
            else:
                quality = "Unhealthy ⛔"
                quality_color = "#eb3349"
                quality_icon = "🔴"
                advice = "Everyone should reduce outdoor activities."
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted PM2.5", f"{sample_pred:.2f} μg/m³")
            col2.metric("Air Quality Level", quality)
            col3.metric("Model Accuracy", 
                       f"{st.session_state.results[model_choice]['metrics']['R² Score']:.2%}")
            
            # Styled advice box
            st.markdown(f"""
            <div style='background: {quality_color}; color: white; padding: 1.5rem; 
                        border-radius: 15px; margin: 1rem 0; text-align: center;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.2);'>
                <p style='font-size: 2rem; margin: 0;'>{quality_icon}</p>
                <p style='font-size: 1.2rem; font-weight: bold; margin: 0.5rem 0;'>{quality}</p>
                <p style='font-size: 1rem; margin: 0;'>{advice}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.success("✅ Prediction completed!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem;'>
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 15px; color: white; 
                box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);'>
        <p style='font-size: 1.5rem; font-weight: bold; margin: 0;'>
            🌍 Beijing PM2.5 Air Quality Prediction System
        </p>
        <p style='font-size: 1.1rem; margin: 1rem 0;'>
            Powered by Machine Learning & Neural Networks
        </p>
        <p style='margin: 0.5rem 0;'>
            🏛️ COMSATS University Islamabad | BS Artificial Intelligence (2024-2028)
        </p>
        <p style='margin: 0.5rem 0;'>
            👩‍💻 Developed by Hasana Zahid & Dur-e-Shahwar
        </p>
        <p style='margin: 1rem 0 0 0; font-size: 0.9rem;'>
            © 2024 All Rights Reserved
        </p>
    </div>
</div>
""", unsafe_allow_html=True)