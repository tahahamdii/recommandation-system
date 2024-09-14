from flask import Flask, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained Random Forest model
rf_model = joblib.load('random_forest_model.pkl')

# Load the dataset
df = pd.read_csv('random_service_dataset.csv')

# Feature engineering (same steps as before)
df['price_per_review'] = df['Base Price'] / (df['Total Reviews'] + 1)
df['price_star_ratio'] = df['Base Price'] / (df['Average Stars'] + 1)
df['availability_numeric'] = df['Availability'].astype(int)

# One-hot encode categorical variables to match the trained model
df_encoded = pd.get_dummies(df, columns=['Title', 'Description', 'Owner'])

# Load the list of model features used during training
model_features = joblib.load('model_features.pkl')  # Ensure this contains the exact feature list

# Reorder columns to match the model's features
df_encoded = df_encoded.reindex(columns=model_features, fill_value=0)

# Remove any columns that shouldn't be present (such as ID and Plus Achetés)
df_encoded = df_encoded.drop(columns=['ID', 'Plus Achetés'], errors='ignore')

@app.route('/')
def home():
    return "Welcome to the Service Purchase Prediction API"

# GET method to retrieve the most purchased service
@app.route('/most_purchased_service', methods=['GET'])
def most_purchased_service():
    try:
        # Make a copy of the encoded DataFrame without the Predicted_Purchases column
        df_for_prediction = df_encoded.copy()

        # Predict purchases for each service using the Random Forest model
        df_for_prediction['Predicted_Purchases'] = rf_model.predict(df_for_prediction)

        # Find the service with the highest predicted purchases
        max_pred_idx = df_for_prediction['Predicted_Purchases'].idxmax()
        most_purchased_service = df.iloc[max_pred_idx]

        # Extract some key characteristics of the most purchased service
        service_info = {
            'Service ID': int(most_purchased_service['ID']),
            'Title': most_purchased_service['Title'],
            'Description': most_purchased_service['Description'],
            'Base Price': float(most_purchased_service['Base Price']),
            'Total Reviews': int(most_purchased_service['Total Reviews']),
            'Average Stars': float(most_purchased_service['Average Stars']),
            'Availability': most_purchased_service['Availability'],
            'Predicted Purchases': float(df_for_prediction.iloc[max_pred_idx]['Predicted_Purchases'])
        }

        # Convert NumPy types to standard Python types
        def convert_types(data):
            if isinstance(data, (pd.Series, pd.DataFrame)):
                data = data.applymap(lambda x: x.item() if hasattr(x, 'item') else x)
            elif isinstance(data, dict):
                for key in data:
                    value = data[key]
                    if isinstance(value, (pd.Series, pd.DataFrame)):
                        data[key] = value.applymap(lambda x: x.item() if hasattr(x, 'item') else x)
                    elif hasattr(value, 'item'):
                        data[key] = value.item()
            return data

        service_info = convert_types(service_info)

        # Return the service details as a JSON response
        return jsonify(service_info)
    except Exception as e:
        # Handle exceptions and return a meaningful error message
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
