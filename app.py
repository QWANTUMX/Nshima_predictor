from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# 1. Load the "Brain" and the "Dictionary"
model = joblib.load('nshima_model.joblib')
model_features = joblib.load('model_features.joblib')

@app.route('/')
def home():
    # This shows your index.html file when you first open the site
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 2. Get data from the HTML form
    portion = float(request.form['portion'])
    food_type = request.form['food_type']
    
    # 3. Prepare the data for the model (The "Switch" logic)
    input_data = pd.DataFrame(columns=model_features)
    input_data.loc[0] = 0
    input_data['Portion_g'] = portion
    
    column_name = f'Staple_Food_{food_type}'
    if column_name in input_data.columns:
        input_data[column_name] = 1
    
    # 4. Make the prediction
    prediction = model.predict(input_data)[0]
    
    # 5. Send the answer back to the HTML
    return render_template('index.html', 
                           prediction_text=f'Predicted Glycemic Load: {prediction:.2f}')

if __name__ == "__main__":
    # Runs the server on your phone
    app.run(host='0.0.0.0', port=5000, debug=True)

