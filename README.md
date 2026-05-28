# 🚕 NYC Taxi Trip Duration Predictor

An end-to-end Big Data engineering and Machine Learning system designed to predict NYC taxi trip durations using distributed data processing and scalable predictive modeling. Built with **PySpark MLlib**, the project integrates large-scale feature engineering, regression modeling, and an interactive Streamlit dashboard for real-time route analysis and trip duration prediction.

---

# 🚀 Project Overview

The system processes over **1.4 million** geospatial and temporal taxi trip records using a distributed PySpark pipeline optimized for scalable analytics and machine learning workflows.

---

# ⚙️ End-to-End Pipeline

```mermaid
graph LR
    A[Data Acquisition] --> B[Exploratory Data Analysis]
    B --> C[Data Cleaning & Preprocessing]
    C --> D[Feature Engineering]
    D --> E[Model Training]
    E --> F[Model Evaluation]
    F --> G[Deployment Dashboard]
```

---

# 📂 Repository Structure

```bash
├── NYC_Taxi_BEST.ipynb      # End-to-end PySpark ML pipeline
├── streamlit_app.py         # Streamlit deployment application
├── eda_plots.png            # Exploratory analysis visualizations
├── evaluation_plots.png     # Prediction error & evaluation charts
├── rmse_mae_comparison.png  # Model metric comparison plots
├── feature_importance.csv   # Ranked feature importance results
└── models/                  # Serialized model assets
    ├── LR_coefficients.npy
    ├── LR_intercept.npy
    ├── scaler_mean.npy
    └── scaler_std.npy
```

---

# 🧹 Data Processing & Cleaning

The preprocessing pipeline performs:

* Removal of invalid ride records
* Filtering anomalous coordinates outside NYC boundaries
* Handling missing and inconsistent values
* Passenger and trip validation checks
* Distributed transformation of spatial-temporal features

---

# 🧠 Feature Engineering

Advanced feature engineering techniques include:

## 📍 Geospatial Features

* Haversine distance calculations
* Manhattan distance estimation
* Pickup and dropoff coordinate extraction

## ⏱️ Temporal Features

* Hour of day
* Day of week
* Month extraction
* Cyclical time encoding for traffic pattern learning

---

# 📊 Machine Learning Models

The project benchmarks multiple distributed regression algorithms using **PySpark MLlib**, including:

* Linear Regression
* Random Forest Regressor
* Gradient-Boosted Trees (GBT) Regressor

The final optimized pipeline focuses on:

* Scalable distributed training
* Regression performance optimization
* Low-latency inference workflows

---

# 📈 Evaluation Metrics

Model performance is evaluated using:

* RMSE (Root Mean Squared Error)
* MAE (Mean Absolute Error)
* R² Score

Evaluation visualizations include:

* Predicted vs Actual plots
* Residual error distributions
* Feature importance analysis

---

# ⚡ Distributed Spark Optimizations

To improve large-scale processing efficiency, the pipeline applies:

* Spark caching strategies
* Memory allocation tuning
* Optimized shuffle partition configurations
* Efficient distributed execution workflows

These optimizations reduce training bottlenecks and improve pipeline scalability.

---

# 🖥️ Streamlit Deployment Dashboard

The project includes an interactive Streamlit application with:

## 📍 Single Trip Prediction

* Custom pickup/dropoff coordinates
* Passenger count selection
* Vendor ID configuration
* Real-time trip duration prediction

## 📊 Batch Prediction Mode

* Upload CSV trip datasets
* Run bulk duration predictions
* Download processed prediction outputs

## 🗺️ Geospatial Visualization

* Interactive Pydeck route mapping
* Coordinate visualization
* Dynamic route analysis

---

# 💡 Mathematical Modeling

## 🌍 Haversine Distance Formula

The system computes spherical distance between coordinates using:

[
d = 2R \arcsin\left(
\sqrt{
\sin^2\left(\frac{\Delta \phi}{2}\right)
+
\cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)
}
\right)
]

---

## 📉 Log Transformation

To stabilize long-tail trip duration distributions:

[
\ln(\text{duration} + 1)
]

This improves regression stability and reduces variance skewness during training.

---

# 🛠️ Installation & Setup

## Prerequisites

* Python 3.9+
* Apache Spark
* Java Runtime Environment

---

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/nyc-taxi-predictor.git
cd nyc-taxi-predictor
```

---

## 2️⃣ Install Dependencies

```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn pyspark pydeck
```

---

## 3️⃣ Configure Spark Environment

Ensure the following environment variables are configured correctly:

* `SPARK_HOME`
* `JAVA_HOME`

---

# ▶️ Running the Application

Launch the Streamlit dashboard locally:

```bash
streamlit run streamlit_app.py
```

---

# 🧰 Technologies Used

## Languages

* Python

## Big Data & Distributed Processing

* PySpark
* Spark MLlib

## Machine Learning

* Scikit-learn
* Gradient-Boosted Trees

## Data Processing

* Pandas
* NumPy

## Visualization

* Matplotlib
* Seaborn
* Pydeck

## Deployment

* Streamlit

---

# 📌 Key Highlights

* Distributed ML pipeline processing 1.4M+ records
* Advanced geospatial and temporal feature engineering
* Optimized Spark execution and caching workflows
* Interactive real-time prediction dashboard
* Batch processing and route visualization support

---

# 📜 License

This project is intended for educational and research purposes.


Train & Test Data: https://drive.google.com/drive/folders/1IBTMt7AmGJyKsHciNP6EduwjW-GRWBxH?usp=sharing
