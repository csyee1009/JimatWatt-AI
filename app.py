from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os

# Configure Flask to look for HTML in the current folder
app = Flask(__name__, template_folder='.')

# --- 1. Load the Model ---
MODEL_FILE = 'energy_model_v1.pkl'

if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
    print(f"✅ Model loaded: {MODEL_FILE}")
else:
    model = None
    print(f"⚠️ ERROR: {MODEL_FILE} not found. Run your notebook first!")

# --- 2. Preprocessing Logic ---
def preprocess_input(data):
    # Convert JSON to DataFrame
    df = pd.DataFrame([data])
    
    # 1. Standardize Column Names
    if 'Day' in df.columns:
        df.rename(columns={'Day': 'Day_of_Week'}, inplace=True)
        
    # 2. Map AC to Binary (Handle 'Yes'/'No' or 1/0)
    val = df['Has_AC'].iloc[0]
    if isinstance(val, str):
        df['Has_AC_Binary'] = 1 if val.lower() == 'yes' else 0
    else:
        df['Has_AC_Binary'] = int(val)
        
    # 3. Create Engineered Features
    df['Is_Weekend'] = 1 if df['Day_of_Week'].iloc[0] in ['Saturday', 'Sunday'] else 0
    df['Temp_x_AC'] = df['Avg_Temperature_C'] * df['Has_AC_Binary']
    
    # 4. ORDERING GUARANTEE
    expected_cols = ['Avg_Temperature_C', 'Household_Size', 'Has_AC_Binary', 'Peak_Hours_Usage_kWh', 'Is_Weekend', 'Temp_x_AC', 'Day_of_Week']
    
    # Return only the columns the model needs, in the right order
    return df[expected_cols]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not found'}), 500

    try:
        data = request.get_json()
        input_df = preprocess_input(data)
        
        # --- A. AI PREDICTION ---
        prediction = model.predict(input_df)[0]
        
        # --- B. PHYSICS ENFORCEMENT (The "Weekend Boost") ---
        if input_df['Is_Weekend'].iloc[0] == 1:
            print("Weekend Detected: Boosting +20%")
            prediction = prediction * 1.20

        # --- C. SAFETY CHECKS ---
        # Ensure prediction is never negative
        prediction = max(prediction, 0.5)
        
        # Define Threshold
        SAFE_THRESHOLD = 20.0
        status = "CRITICAL" if prediction > SAFE_THRESHOLD else "NORMAL"
        
        return jsonify({
            'prediction': round(prediction, 2),
            'status': status,
            'threshold': SAFE_THRESHOLD
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)