# ML Project for Lung Cancer Prediction

This project aims to predict lung cancer based on various health indicators using a RandomForestClassifier model.

## Project Structure
- `data/`: Contains preprocessed datasets.
- `models/`: Stores trained machine learning models.
- `notebooks/`: Jupyter notebooks for data exploration, preprocessing, and model training.
- `src/`: Source code for prediction scripts and utility functions.

## Setup Instructions
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run Prediction
To make a prediction using the trained model, run the `predict_model_1.py` script:

```bash
python src/predict_model_1.py
```

This script loads the pre-trained model and makes a prediction on a sample data point.
