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

# Reorder columns to match the model's features
model_features = joblib.load('model_features.pkl')  # Load saved model features
df_encoded = df_encoded.reindex(columns=model_features, fill_value=0)


@app.route('/')
def home():
    return "Welcome to the Service Purchase Prediction API"


# GET method to retrieve the most purchased service
@app.route('/most_purchased_service', methods=['GET'])
def most_purchased_service():
    # Predict purchases for each service using the Random Forest model
    df_encoded['Predicted_Purchases'] = rf_model.predict(df_encoded)

    # Find the service with the highest predicted purchases
    max_pred_idx = df_encoded['Predicted_Purchases'].idxmax()
    most_purchased_service = df.iloc[max_pred_idx]

    # Extract some key characteristics of the most purchased service
    service_info = {
        'Service ID': most_purchased_service['ID'],
        'Title': most_purchased_service['Title'],
        'Description': most_purchased_service['Description'],
        'Base Price': most_purchased_service['Base Price'],
        'Total Reviews': most_purchased_service['Total Reviews'],
        'Average Stars': most_purchased_service['Average Stars'],
        'Availability': most_purchased_service['Availability'],
        'Predicted Purchases': df_encoded.iloc[max_pred_idx]['Predicted_Purchases']
    }

    # Return the service details as a JSON response
    return jsonify(service_info)


if __name__ == '__main__':
    app.run(debug=True)
