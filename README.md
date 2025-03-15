# Phishing Detection System

A machine learning-based phishing detection system that analyzes URLs to determine whether they are legitimate or phishing attempts. The model is trained using symbolic rules and a neural network for better accuracy.

## Features
- Uses a trained deep learning model for phishing detection
- Includes a whitelist check for trusted domains
- Preprocessing and feature extraction from URLs
- Symbolic rule optimization using a genetic algorithm
- Interactive command-line interface for URL checking

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.x
- TensorFlow
- NumPy
- Scikit-learn
- Pandas
- TLDExtract

### Install Dependencies
Run the following command to install necessary Python packages:
```sh
pip install tensorflow numpy scikit-learn pandas tldextract
```

## Usage
### 1. Train the Model (if not already trained)
If a pre-trained model is not found, the script will automatically train a new model using the `dataset_phishing.csv` file.
```sh
python phishing_detection.py
```

### 2. Checking URLs
Once the model is trained or loaded, you can check URLs interactively:
```sh
Enter a URL to check (or 'quit' to exit): https://example.com
Prediction: Legitimate
Confidence: 95.78%
```

## Project Structure
```
├── phishing_detection.py  # Main script
├── dataset_phishing.csv   # Training dataset
├── phishing_model.h5      # Trained model
├── phishing_scaler.pkl    # Scaler for feature normalization
├── phishing_features.pkl  # Feature names
├── phishing_rule_weights.pkl  # Rule weights for detection
├── phishing_threshold.pkl  # Detection threshold
└── README.md              # Project documentation
```

## Model Components
- **Neural Network Model**: A trained deep learning model to classify phishing URLs.
- **Symbolic Rules**: A set of predefined rules optimized using a genetic algorithm.
- **Scaler**: Used to normalize input features for consistent model performance.

## Future Improvements
- Enhance rule optimization for better phishing detection
- Deploy as a web API for real-time URL scanning
- Expand dataset for improved generalization
