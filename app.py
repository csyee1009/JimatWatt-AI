from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__, template_folder='.')

MODEL_FILE = 'energy_model_v1.pkl'

# Load Model
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
    print(f"✅ Model loaded: {MODEL_FILE}")
else:
    model = None

def preprocess_input(data):
    df = pd.DataFrame([data])
    
    if 'Day' in df.columns:
        df.rename(columns={'Day': 'Day_of_Week'}, inplace=True)
        
    val = df['Has_AC'].iloc[0]
    if isinstance(val, str):
        df['Has_AC_Binary'] = 1 if val.lower() == 'yes' else 0
    else:
        df['Has_AC_Binary'] = int(val)
        
    df['Is_Weekend'] = 1 if df['Day_of_Week'].iloc[0] in ['Saturday', 'Sunday'] else 0
    df['Temp_x_AC'] = df['Avg_Temperature_C'] * df['Has_AC_Binary']
    
    expected_cols = ['Avg_Temperature_C', 'Household_Size', 'Has_AC_Binary', 'Peak_Hours_Usage_kWh', 'Is_Weekend', 'Temp_x_AC', 'Day_of_Week']
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
        
        # Base AI Prediction
        consumption = model.predict(input_df)[0]
        
        # Logic 1: Weekend Boost (People are home)
        if input_df['Is_Weekend'].iloc[0] == 1:
            consumption = consumption * 1.15 # 15% increase on weekends

        # Logic 2: AC Temperature Scaling
        # If AC is ON, we ensure consumption rises as Temp rises
        if input_df['Has_AC_Binary'].iloc[0] == 1:
            temp = input_df['Avg_Temperature_C'].iloc[0]
            
            # Base penalty for just having AC on
            ac_multiplier = 1.10 
            
            # Additional penalty: For every degree above 25°C, add 2% extra usage
            if temp > 25.0:
                extra_load = (temp - 25.0) * 0.02
                ac_multiplier += extra_load
                
            consumption = consumption * ac_multiplier

        # Ensure prediction is never negative
        consumption = round(max(consumption, 0.5), 2)

        # 2. Check for Solar Option
        has_solar = data.get('Has_Solar', False)
        
        if has_solar:
            # SOLAR LOGIC
            solar_size = float(data.get('Solar_Panel_Size_kW', 0))
            
            # Physics Formula: Size * 5 Sun Hours * 0.85 Efficiency
            generation = round(solar_size * 5.0 * 0.85, 2)
            
            net_energy = round(generation - consumption, 2)
            bill_impact = round(net_energy * 0.218, 2) # Updated to approx RM tariff (0.218 / kWh)
            
            return jsonify({
                'mode': 'solar',
                'consumption': consumption,
                'generation': generation,
                'net_energy': net_energy,
                'bill_impact': bill_impact
            })
            
        else:
            # STANDARD LOGIC (No Solar)
            SAFE_THRESHOLD = 20.0
            status = "CRITICAL" if consumption > SAFE_THRESHOLD else "NORMAL"
            
            return jsonify({
                'mode': 'standard',
                'prediction': consumption,
                'status': status,
                'threshold': SAFE_THRESHOLD
            })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)