# Electric Load Forecasting Application

This is a machine learning-powered web application that provides accurate electricity load forecasting for 10 major US cities. The application leverages weather patterns, historical consumption data, and advanced predictive modeling to estimate electricity demand, aiding in energy management and grid optimization.

## Features

### Load Prediction Models
- **XGBoost**: Advanced gradient boosting model optimized for precise, non-linear predictions.
- **Random Forest**: Ensemble-based approach ensuring robust and stable load forecasting.
- **Naive Baseline**: A simple lag-based prediction model used for benchmarking performance.

### Interactive Cluster Analysis
- Interactive K-Means clustering visualization mapping data points by weather and demand.
- Dynamic adjustments for the number of clusters (k = 1 to 10).
- Identifies underlying consumption behaviors and weather relationships across different cities.

## Data & Features
The application performs predictions using a combination of rich meteorological and temporal features:
- **Weather Metrics**: Temperature, Apparent Temperature, Dew Point, Humidity, Pressure, Precipitation Intensity/Accumulation, Ozone levels.
- **Temporal Data**: Hour of the day, Day of the week, Month.
- **Derived Features**: Precipitation type (One-Hot Encoded) and precomputed K-Means cluster assignments.

## Tech Stack
- **Backend**: Python, Flask
- **Machine Learning**: Scikit-Learn, XGBoost, Pandas
- **Frontend / Visualization**: Chart.js (Predictions), Plotly (Clustering), Bootstrap (Responsive UI)

## How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/tahabintariq/electricity-load-predictor.git
   cd electricity-load-predictor
   ```

2. **Install requirements:**
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Ensure Flask, pandas, joblib, and other required ML libraries are installed).*

3. **Start the application:**
   ```bash
   python app.py
   ```

4. **Access the Web App:**
   Open your browser and navigate to `http://localhost:5000`

## Usage
1. Navigate to the **Load Prediction** tab.
2. Select a city (e.g., Dallas, NYC, Seattle).
3. Choose a prediction model and a date range (between `2018-07-01` and `2020-05-23`).
4. Click "Predict" to visualize the predicted vs. actual electricity demand on the interactive chart.
5. Explore the **Cluster Analysis** tab to adjust K-values and view patterns using PCA coordinates.

