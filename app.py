from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import os

app = Flask(__name__)

# Global variables to store the scaler and data
scaler = None
time_step = 60
data_values = None
model = None  # Ensure global access to the model

def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global data_values, scaler, model  # Ensure model is accessible here

    # Retrieve uploaded files
    csv_file = request.files.get('csv_file')
    model_file = request.files.get('model_file')

    # Process the uploaded model file
    if model_file and model_file.filename.endswith('.h5'):
        try:
            # Save the uploaded .h5 file to a temporary location
            model_path = os.path.join('static', model_file.filename)
            model_file.save(model_path)
            
            # Load the pre-trained model from the saved file
            model = load_model(model_path)
            print("Model loaded successfully!")  # Debug print
            
            # Optionally, remove the file after loading the model to clean up
            os.remove(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")  # Print the error for debugging
            return redirect(url_for('index'))  # If loading fails, redirect back

    else:
        return redirect(url_for('index'))  # If no model is uploaded, redirect back

    # Process the uploaded CSV file if present
    if csv_file and csv_file.filename.endswith('.csv'):
        try:
            # Read and process the CSV data
            data = pd.read_csv(csv_file, date_parser=True)
            data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
            data.set_index('Date', inplace=True)
            data = data[['Close']]

            # Scale the data
            data_values = data.values
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data_values)

            # Create test dataset
            X, y = create_dataset(scaled_data, time_step)
            X = X.reshape((X.shape[0], X.shape[1], 1))

            # Check if the model is loaded
            if model is None:
                print("Model not loaded!")  # Debug info
                return redirect(url_for('index'))

            # Make predictions
            predictions = model.predict(X)
            predictions = scaler.inverse_transform(predictions)

            # Plot actual vs predicted prices
            plt.figure(figsize=(14, 5))
            plt.plot(data.index, data_values, label='True Price', color='blue')
            predicted_dates = data.index[-len(predictions):]
            plt.plot(predicted_dates, predictions, label='Predicted Price', color='red')
            plt.gca().xaxis.set_major_formatter(DateFormatter('%Y'))
            plt.xlabel('Year')
            plt.ylabel('Stock Price (USD)')
            plt.title('Stock Price Prediction')
            plt.legend()
            plt.grid(True)
            plt.gcf().autofmt_xdate()

            # Ensure the static folder exists
            static_folder = os.path.join(os.getcwd(), 'static')
            if not os.path.exists(static_folder):
                os.makedirs(static_folder)
                print("Static folder created")  # Debugging print

            # Save the plot to a file named 'prediction.png' in the static folder
            img_path = os.path.join(static_folder, 'prediction.png')

            # Overwrite the existing prediction.png file
            plt.savefig(img_path)
            plt.close()

            # Check if the file was saved correctly
            if os.path.exists(img_path):
                print(f"Image saved successfully at {img_path}")  # Debugging print
            else:
                print("Image not saved!")  # Debugging print

            # Reload the page with the updated image
            return render_template('index.html', image_url=url_for('static', filename='prediction.png'))

        except Exception as e:
            print(f"Error processing CSV: {e}")  # Debug information for error
            return redirect(url_for('index'))

    return redirect(url_for('index'))

if __name__ == '__main__':
    # Ensure the static folder exists to store the plot image
    static_folder = os.path.join(os.getcwd(), 'static')
    if not os.path.exists(static_folder):
        os.makedirs(static_folder)
    app.run(debug=True)
