from flask import Flask, render_template, request, jsonify
import pandas as pd
from datetime import datetime
import joblib
import os
import json

app = Flask(__name__)

# List of cities and models
CITIES = ['dallas', 'houston', 'la', 'nyc', 'philadelphia', 
          'phoenix', 'san_antonio', 'san_diego', 'san_jose', 'seattle']
MODELS = ['xgb_model', 'rf_model', 'naive_model']

# Date range constants (removed time component)
MIN_DATE = "2018-07-01"
MAX_DATE = "2020-05-23"

# Define the features used for prediction
FEATURES = [
    'temperature', 'apparentTemperature', 'dewPoint', 'humidity', 
    'pressure', 'precipIntensity', 'ozone', 'precipAccumulation',
    'hour', 'day_of_week', 'month', 'precipType', 'kmeans_cluster'
]

def prepare_features(df):
    """Prepare features with correct one-hot encoding"""
    features = df[FEATURES].copy()
    
    # Create one-hot encoding with all possible categories
    precip_types = ['rain', 'snow', 'No Precipitation']
    
    # Create dummy columns for precipType
    dummies = pd.get_dummies(features['precipType'], prefix='precipType')
    
    # Add missing columns with 0s
    for precip_type in precip_types:
        col_name = f'precipType_{precip_type}'
        if col_name not in dummies.columns:
            dummies[col_name] = 0
    
    # Drop original precipType column and join with dummies
    features = features.drop('precipType', axis=1)
    features = pd.concat([features, dummies[['precipType_rain', 'precipType_snow']]], axis=1)
    
    return features

def naive_forecast(data, lag_period):
    """Generate naive forecast using lag period"""
    return data['demand'].shift(lag_period)

@app.route('/get_cluster_viz', methods=['POST'])
def get_cluster_viz():
    try:
        # Get k value from request
        k_value = int(request.form['k_value'])
        
        # Load the precomputed cluster results
        with open('cluster_results.json', 'r') as f:
            cluster_data = json.load(f)
        
        # Get PCA coordinates and cluster labels for selected k
        pca_coords = cluster_data['pca_coords']
        cluster_labels = cluster_data[f'k{k_value}_labels']
        
        # Prepare data for plotting
        plot_data = {
            'pca_coords': pca_coords,
            'cluster_labels': cluster_labels,
            'k_value': k_value
        }
        
        return jsonify(plot_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/')
def home():
    return render_template('index.html', 
                         cities=CITIES,
                         models=MODELS,
                         min_date=MIN_DATE,
                         max_date=MAX_DATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        city = request.form['city']
        model_type = request.form['model']
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        lag_period = int(request.form.get('lag_period', 24))  # Default to 24 if not provided

        # Load city data
        city_data = pd.read_csv(f'data/{city}.csv')
        
        # Convert timestamp to datetime
        city_data['time'] = pd.to_datetime(city_data['time'], unit='s')
        
        # Convert input dates to datetime (assuming they come as YYYY-MM-DD)
        start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')
        end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')
        
        # Filter data by date range (using date component only)
        mask = (city_data['time'].dt.strftime('%Y-%m-%d') >= start_date) & \
               (city_data['time'].dt.strftime('%Y-%m-%d') <= end_date)
        filtered_data = city_data[mask]

        # Make predictions based on model type
        if model_type == 'naive_model':
            # Sort data by time to ensure correct lag calculation
            city_data = city_data.sort_values('time')
            predictions = naive_forecast(city_data, lag_period)
            # Filter predictions to match the selected date range
            predictions = predictions[mask]
        else:
            # Load the appropriate model
            model = joblib.load(f'models/{model_type}.pkl')
            # Prepare features with correct encoding
            features = prepare_features(filtered_data)
            predictions = model.predict(features)

        # Check if start_date equals end_date
        same_day = start_date == end_date
        
        # Prepare results with appropriate date format
        if same_day:
            dates = filtered_data['time'].dt.strftime('%H:00').tolist()
        else:
            dates = filtered_data['time'].dt.strftime('%Y-%m-%d').tolist()

        results = {
            'predictions': predictions.tolist(),
            'actual': filtered_data['demand'].tolist(),
            'dates': dates,
            'same_day': same_day
        }

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)