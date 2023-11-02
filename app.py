from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import pandas as pd
import os
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import plotly
from plotly.tools import mpl_to_plotly
import matplotlib.pyplot as plt
import numpy as np
import random
import plotly.express as px
import json


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
upload_folder_path = os.path.join(os.getcwd(), app.config['UPLOAD_FOLDER'])
os.makedirs(upload_folder_path, exist_ok=True)

# AnalystAgent class with data processing, training, and evaluation methods
class AnalystAgent:
    def __init__(self):
        self.models = {}
        self.scalers = {}

    def load_and_preprocess_data(self, file_path, key, target_variable):
        df = pd.read_csv(file_path)
        df.dropna(inplace=True)

        # Split the data into features and target
        X = df.drop(target_variable, axis=1)
        y = df[target_variable]

        # Handle categorical features
        X = pd.get_dummies(X)

        # Scaling numerical features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        self.scalers[key] = scaler

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.models[key] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'model': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        self.models[key]['model'].fit(X_train, y_train)

    def predict(self, key, input_data):
        model_info = self.models[key]
        model = model_info['model']
        scaler = self.scalers[key]

        # Preprocess input_data using the same steps as the training data
        input_data_processed = pd.get_dummies(pd.DataFrame(input_data))
        input_data_processed = scaler.transform(input_data_processed)

        # Ensure the input data has the same number of columns as the training data
        input_data_processed = input_data_processed.reindex(
            columns=model_info['X_train'].columns, fill_value=0
        )

        return model.predict(input_data_processed)

    def evaluate_model(self, key):
        model_info = self.models[key]
        predictions = model_info['model'].predict(model_info['X_test'])
        mae = mean_absolute_error(model_info['y_test'], predictions)
        
        # Calculate SHAP values for the test set
        explainer = shap.TreeExplainer(model_info['model'])
        shap_values = explainer.shap_values(model_info['X_test'])

        # Create SHAP summary plot
        shap.summary_plot(shap_values, model_info['X_test'], show=False)
        shap_fig = plt.gcf()
        shap_plotly_fig = mpl_to_plotly(shap_fig)

        return {
            'mae': mae,
            'shap_plot': plotly.io.to_json(shap_plotly_fig)
        }
    

# Function to generate random data based on industry
def generate_random_data(num_samples=100, industry=None):
    # Define different features for different industries
    industry_features = {
        'healthcare': ['patient_satisfaction', 'treatment_success', 'operational_efficiency'],
        'finance': ['market_volatility', 'client_transactions', 'interest_rates'],
        'sports': ['team_effort', 'fan_engagement', 'sponsor_revenue'],
    }

    # If no industry is given, select a random one
    if not industry:
        industry = random.choice(list(industry_features.keys()))

    # Generate random data for the selected industry
    features = industry_features[industry]
    data = {
        feature: np.random.rand(num_samples).tolist() for feature in features
    }
    data['strategyInsights'] = (np.random.rand(num_samples) * 100).tolist()

    return pd.DataFrame(data), industry

@app.route('/')
def index():
    # Placeholder for the main page rendering
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('file')
    if not files:
        return jsonify(message="No files provided"), 400

    agent = AnalystAgent()
    results = {}

    for file in files:
        filename = secure_filename(file.filename)
        if not filename.endswith('.csv'):
            return jsonify(message=f"Incorrect file type for {filename}"), 400

        # Assume the target variable is the last column for simplicity
        target_variable = 'target'  # Placeholder for actual target variable name

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Load, preprocess, and train the model
        agent.load_and_preprocess_data(file_path, filename, target_variable)

        # Evaluate the model and get SHAP values
        evaluation_results = agent.evaluate_model(filename)
        results[filename] = {
            'performance': evaluation_results['mae'],
            'shap_plot': evaluation_results['shap_plot']  # Send SHAP values visualization
        }

    return jsonify(results), 200

@app.route('/generate', methods=['GET'])
def generate_data():
    industry = request.args.get('industry')
    df_random_data, industry = generate_random_data(industry=industry)

    # Create a plot from the generated data
    fig = px.scatter(df_random_data, x=df_random_data.columns[0], y='strategyInsights',
                     title=f'Strategy Insights for {industry.capitalize()} Industry')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Provide a basic explanation based on industry
    explanations = {
        'healthcare': 'Random healthcare industry data, including patient feedback, treatment outcomes, and operational metrics.',
        'finance': 'Random financial industry data, reflecting market conditions, client activity, and interest rates.',
        'sports': 'Random sports industry data, with metrics on team performance, fan engagement, and revenue.'
    }
    explanation = explanations.get(industry, 'Random industry data generated.')

    return jsonify(result=graphJSON, explanation=explanation), 200

@app.route('/process_buyers_customers', methods=['POST'])
def process_buyers_customers():
    # Check if fileId is in the form data
    if 'fileId' not in request.form:
        return jsonify(message="No file identifier provided"), 400

    # Get the fileId from the form data
    file_id = request.form['fileId']

    # Based on fileId, determine the path to the file on the server
    if file_id == 'file1':
        file_path = '/mnt/data/Buyers_Customers - Sheet1.csv'  # Update the path to your actual file
    else:
        return jsonify(message="Invalid file identifier"), 400

    agent = AnalystAgent()
    # Assuming the target variable is the last column
    target_variable = 'strategyInsights'  # Adjust to the correct target variable name
    agent.load_and_preprocess_data(file_path, 'buyers_customers', target_variable)
    evaluation_results = agent.evaluate_model('buyers_customers')

    return jsonify({
        'performance': evaluation_results['mae'],
        'shap_plot': evaluation_results['shap_plot']
    }), 200


@app.route('/process_sales_revenue', methods=['POST'])
def process_sales_revenue():
    # Check if fileId is in the form data
    if 'fileId' not in request.form:
        return jsonify(message="No file identifier provided"), 400

    # Get the fileId from the form data
    file_id = request.form['fileId']

    # Based on fileId, determine the path to the file on the server
    if file_id == 'file2':
        file_path = '/mnt/data/Sales_Revenue - Sheet1.csv'  # Path to the uploaded sales revenue file
    else:
        return jsonify(message="Invalid file identifier"), 400

    agent = AnalystAgent()
    # Assuming the target variable is named 'Revenue' in your CSV, replace it with the actual name
    target_variable = 'Revenue'  # Replace with the actual target variable column name
    agent.load_and_preprocess_data(file_path, 'sales_revenue', target_variable)
    evaluation_results = agent.evaluate_model('sales_revenue')

    return jsonify({
        'performance': evaluation_results['mae'],
        'shap_plot': evaluation_results['shap_plot']
    }), 200


if __name__ == '__main__':
    app.run(debug=True)