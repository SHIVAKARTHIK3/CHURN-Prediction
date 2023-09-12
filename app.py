from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

loaded_model = pickle.load(open('model.churn', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))  # Load the scaler

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        user_input = request.form.to_dict()
        
        # Preprocess the user input
        user_df = pd.DataFrame(user_input, index=[0])
        user_features = user_df[X_final.columns]  # Assuming X_final.columns matches input fields
        user_features_scaled = scaler.transform(user_features)
        
        # Make predictions using the loaded model
        prediction = loaded_model.predict(user_features_scaled)
        result = 'Churn' if prediction[0] == 1 else 'Not Churn'
        
        return render_template('index.html', prediction_result=result)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)


