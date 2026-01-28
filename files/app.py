from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os

app = Flask(__name__)

# Global variables for model and encoders
model = None
encoders = {}
feature_columns = []
original_df = None

def load_and_train_model():
    """Load data and train the Random Forest model"""
    global model, encoders, feature_columns, original_df
    
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv('Indian_Airlines.csv')
    
    # Store original dataframe for API endpoints
    original_df = df.copy()
    
    print(f"Dataset loaded: {len(df)} rows")
    
    # Drop the unnamed column if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    # Drop flight number as it's not useful for prediction
    if 'flight' in df.columns:
        df = df.drop(columns=['flight'])
    
    # Check for missing values
    print("\nChecking for missing values...")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("Missing values found:")
        print(missing[missing > 0])
        # Drop rows with missing values
        df = df.dropna()
        print(f"After removing missing values: {len(df)} rows")
    
    # Create price categories
    print("\nCreating price categories...")
    df['price_category'] = pd.cut(df['price'], 
                                   bins=[0, 3000, 7000, np.inf], 
                                   labels=['Low', 'Medium', 'High'])
    
    # Check if price_category has NaN
    if df['price_category'].isnull().sum() > 0:
        print(f"Warning: {df['price_category'].isnull().sum()} rows have invalid price categories")
        df = df.dropna(subset=['price_category'])
        print(f"After removing invalid price categories: {len(df)} rows")
    
    # Encode categorical features
    categorical_columns = ['airline', 'source_city', 'destination_city', 
                          'departure_time', 'arrival_time', 'stops', 'class']
    
    print("\nEncoding categorical variables...")
    encoders = {}
    for col in categorical_columns:
        if col in df.columns:
            encoders[col] = LabelEncoder()
            # Handle any potential string issues
            df[col] = df[col].astype(str)
            df[col] = encoders[col].fit_transform(df[col])
            print(f"  {col}: {len(encoders[col].classes_)} unique values")
    
    # Prepare features and target
    X = df.drop('price_category', axis=1)
    y = df['price_category']
    
    feature_columns = X.columns.tolist()
    print(f"\nFeatures: {feature_columns}")
    
    # Check for any remaining NaN values
    if X.isnull().sum().sum() > 0:
        print("\nWarning: NaN values still present in features:")
        print(X.isnull().sum()[X.isnull().sum() > 0])
        X = X.fillna(0)  # Fill remaining NaN with 0
    
    if y.isnull().sum() > 0:
        print(f"\nWarning: {y.isnull().sum()} NaN values in target variable")
        # Remove rows where target is NaN
        mask = ~y.isnull()
        X = X[mask]
        y = y[mask]
    
    print(f"\nFinal dataset size: {len(X)} rows")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {len(X_train)} rows")
    print(f"Test set: {len(X_test)} rows")
    
    # Train Random Forest model
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f"✓ Model trained successfully!")
    print(f"✓ Model accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return df

@app.route('/')
def index():
    """Serve the main application page"""
    # For now, return a simple message since we don't have a templates folder
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Indian Airlines API</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background: #f5f5f5;
            }
            h1 { color: #333; }
            .endpoint {
                background: white;
                padding: 15px;
                margin: 10px 0;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            code {
                background: #f0f0f0;
                padding: 2px 6px;
                border-radius: 3px;
            }
        </style>
    </head>
    <body>
        <h1>🛫 Indian Airlines ML API</h1>
        <p>The Flask backend is running successfully!</p>
        
        <h2>Available API Endpoints:</h2>
        
        <div class="endpoint">
            <h3>GET /api/stats</h3>
            <p>Get dataset statistics</p>
            <code>http://localhost:5000/api/stats</code>
        </div>
        
        <div class="endpoint">
            <h3>GET /api/data</h3>
            <p>Get paginated flight data</p>
            <code>http://localhost:5000/api/data?page=1&per_page=20</code>
        </div>
        
        <div class="endpoint">
            <h3>GET /api/analysis</h3>
            <p>Get analysis data for visualizations</p>
            <code>http://localhost:5000/api/analysis</code>
        </div>
        
        <div class="endpoint">
            <h3>POST /api/predict</h3>
            <p>Predict flight price</p>
            <p>Send JSON with flight details</p>
        </div>
        
        <p style="margin-top: 30px; color: #666;">
            💡 <strong>Tip:</strong> Use the <code>indian_airlines_app.html</code> file to access the full web interface.
        </p>
    </body>
    </html>
    """

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for price prediction"""
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['airline', 'source_city', 'destination_city', 
                          'departure_time', 'arrival_time', 'stops', 
                          'class', 'duration', 'days_left']
        
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Create a DataFrame with the input
        input_data = pd.DataFrame([{
            'airline': str(data['airline']),
            'source_city': str(data['source_city']),
            'destination_city': str(data['destination_city']),
            'departure_time': str(data['departure_time']),
            'arrival_time': str(data['arrival_time']),
            'stops': str(data['stops']),
            'class': str(data['class']),
            'duration': float(data['duration']),
            'days_left': int(data['days_left']),
            'price': 0  # Placeholder
        }])
        
        # Encode categorical variables
        for col in ['airline', 'source_city', 'destination_city', 
                   'departure_time', 'arrival_time', 'stops', 'class']:
            if col in encoders:
                try:
                    input_data[col] = encoders[col].transform(input_data[col])
                except ValueError as e:
                    # Handle unknown categories - use the most common value
                    print(f"Warning: Unknown value for {col}: {data[col]}")
                    input_data[col] = 0
        
        # Make prediction
        X_pred = input_data[feature_columns]
        prediction = model.predict(X_pred)[0]
        probabilities = model.predict_proba(X_pred)[0]
        
        # Estimate actual price based on prediction and features
        category_prices = {
            'Low': 2000,
            'Medium': 5000,
            'High': 9000
        }
        
        base_price = category_prices.get(prediction, 5000)
        
        # Add variation based on features
        duration = float(data['duration'])
        days_left = int(data['days_left'])
        
        estimated_price = base_price
        estimated_price += duration * 200
        
        if days_left <= 3:
            estimated_price += 2000
        elif days_left <= 7:
            estimated_price += 1000
        elif days_left > 30:
            estimated_price -= 500
        
        if data['stops'] == 'one':
            estimated_price += 500
        elif data['stops'] == 'two_or_more':
            estimated_price += 1000
            
        if data['class'] == 'Business':
            estimated_price *= 2.5
        
        # Add small random variation
        estimated_price += np.random.randint(-200, 200)
        estimated_price = max(1000, int(estimated_price))  # Minimum price
        
        return jsonify({
            'success': True,
            'category': prediction,
            'estimated_price': int(estimated_price),
            'confidence': {
                'Low': float(probabilities[0]),
                'Medium': float(probabilities[1]),
                'High': float(probabilities[2])
            }
        })
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get dataset statistics"""
    try:
        df = original_df.copy()
        
        # Remove any rows with NaN in critical columns
        df = df.dropna(subset=['price', 'airline'])
        
        stats = {
            'total_flights': int(len(df)),
            'avg_price': float(df['price'].mean()),
            'min_price': float(df['price'].min()),
            'max_price': float(df['price'].max()),
            'unique_airlines': int(df['airline'].nunique()),
            'unique_routes': int(df[['source_city', 'destination_city']].drop_duplicates().shape[0]),
            'airlines': sorted(df['airline'].dropna().unique().tolist()),
            'cities': sorted(list(set(
                df['source_city'].dropna().unique().tolist() + 
                df['destination_city'].dropna().unique().tolist()
            ))),
            'departure_times': sorted(df['departure_time'].dropna().unique().tolist()),
            'arrival_times': sorted(df['arrival_time'].dropna().unique().tolist())
        }
        
        return jsonify(stats)
        
    except Exception as e:
        print(f"Error in get_stats: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 400

@app.route('/api/data', methods=['GET'])
def get_data():
    """Get paginated flight data"""
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        
        df = original_df.copy()
        df = df.dropna()  # Remove rows with any NaN
        
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        
        # Convert to dict and handle NaN
        data = df.iloc[start_idx:end_idx].fillna('N/A').to_dict('records')
        
        return jsonify({
            'data': data,
            'total': len(df),
            'page': page,
            'per_page': per_page,
            'total_pages': (len(df) + per_page - 1) // per_page
        })
        
    except Exception as e:
        print(f"Error in get_data: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 400

@app.route('/api/analysis', methods=['GET'])
def get_analysis():
    """Get analysis data for visualizations"""
    try:
        df = original_df.copy()
        df = df.dropna(subset=['price'])  # Only drop rows without price
        
        # Average price by airline
        airline_avg = df.groupby('airline')['price'].mean().to_dict()
        airline_avg = {k: float(v) for k, v in airline_avg.items()}
        
        # Average price by class
        class_avg = df.groupby('class')['price'].mean().to_dict()
        class_avg = {k: float(v) for k, v in class_avg.items()}
        
        # Average price by stops
        stops_avg = df.groupby('stops')['price'].mean().to_dict()
        stops_avg = {k: float(v) for k, v in stops_avg.items()}
        
        # Average price by source city
        source_avg = df.groupby('source_city')['price'].mean().to_dict()
        source_avg = {k: float(v) for k, v in source_avg.items()}
        
        # Average price by destination city
        dest_avg = df.groupby('destination_city')['price'].mean().to_dict()
        dest_avg = {k: float(v) for k, v in dest_avg.items()}
        
        return jsonify({
            'airline_prices': airline_avg,
            'class_prices': class_avg,
            'stops_prices': stops_avg,
            'source_prices': source_avg,
            'destination_prices': dest_avg
        })
        
    except Exception as e:
        print(f"Error in get_analysis: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 400

if __name__ == '__main__':
    print("=" * 60)
    print("🛫 INDIAN AIRLINES ML API SERVER")
    print("=" * 60)
    print("\nLoading and training model...")
    
    try:
        load_and_train_model()
        print("\n" + "=" * 60)
        print("✓ Server is ready!")
        print("=" * 60)
        print("\n📍 Access the API at: http://localhost:5000")
        print("📍 For full web interface, open: indian_airlines_app.html")
        print("\nPress Ctrl+C to stop the server\n")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ ERROR STARTING SERVER")
        print("=" * 60)
        print(f"\nError: {str(e)}")
        print("\nPlease check:")
        print("1. Indian_Airlines.csv is in the same folder as app.py")
        print("2. The CSV file is not corrupted")
        print("3. All required packages are installed")
        import traceback
        traceback.print_exc()
