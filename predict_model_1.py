
import pickle
import pandas as pd

def load_model(model_path='models/model_1.pkl'):
    """Loads the trained machine learning model."""
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def make_prediction(model, data_point_df):
    """
    Makes a prediction using the loaded model.
    data_point_df should be a pandas DataFrame with the same features as the training data.
    """
    prediction = model.predict(data_point_df)
    return prediction

if __name__ == "__main__":
    print("Loading model for prediction...")
    try:
        loaded_model = load_model()
        print("Model loaded successfully.")

        feature_names_expected = []
        if hasattr(loaded_model, 'feature_names_in_') and loaded_model.feature_names_in_ is not None:
            feature_names_expected = loaded_model.feature_names_in_.tolist()
            print(f"Model expects features: {feature_names_expected}")
        else:
            # Fallback if feature_names_in_ is not available or None.
            # This hardcoded list must match the columns of the X DataFrame used during training.
            feature_names_expected = ['AGE', 'GENDER', 'SMOKING', 'FINGER_DISCOLORATION', 'MENTAL_STRESS',
                                      'EXPOSURE_TO_POLLUTION', 'LONG_TERM_ILLNESS', 'ENERGY_LEVEL',
                                      'IMMUNE_WEAKNESS', 'BREATHING_ISSUE', 'ALCOHOL_CONSUMPTION',
                                      'THROAT_DISCOMFORT', 'OXYGEN_SATURATION', 'CHEST_TIGHTNESS',
                                      'FATIGUE', 'COUGHING', 'WHEEZING', 'SWALLOWING_DIFFICULTY',
                                      'ALLERGY', 'DRY_COUGH', 'FAMILY_HISTORY', 'SMOKING_FAMILY_HISTORY', 'STRESS_IMMUNE']
            print(f"Using fallback feature names: {feature_names_expected}")

        # Create a base dictionary for dummy data with default values (e.g., 0)
        # This ensures all expected features are present initially.
        dummy_data_dict = {col: [0] for col in feature_names_expected}

        # Override specific values for demonstration purposes
        # Ensure these are for features actually in feature_names_expected and encoded correctly.
        if 'AGE' in dummy_data_dict: dummy_data_dict['AGE'] = [60]
        if 'GENDER' in dummy_data_dict: dummy_data_dict['GENDER'] = [1] # Assuming male, already encoded
        if 'SMOKING' in dummy_data_dict: dummy_data_dict['SMOKING'] = [1]
        if 'FINGER_DISCOLORATION' in dummy_data_dict: dummy_data_dict['FINGER_DISCOLORATION'] = [0]
        if 'MENTAL_STRESS' in dummy_data_dict: dummy_data_dict['MENTAL_STRESS'] = [1]
        if 'EXPOSURE_TO_POLLUTION' in dummy_data_dict: dummy_data_dict['EXPOSURE_TO_POLLUTION'] = [0]
        if 'LONG_TERM_ILLNESS' in dummy_data_dict: dummy_data_dict['LONG_TERM_ILLNESS'] = [1]
        if 'ENERGY_LEVEL' in dummy_data_dict: dummy_data_dict['ENERGY_LEVEL'] = [0]
        if 'IMMUNE_WEAKNESS' in dummy_data_dict: dummy_data_dict['IMMUNE_WEAKNESS'] = [1]
        if 'BREATHING_ISSUE' in dummy_data_dict: dummy_data_dict['BREATHING_ISSUE'] = [1]
        if 'ALCOHOL_CONSUMPTION' in dummy_data_dict: dummy_data_dict['ALCOHOL_CONSUMPTION'] = [0]
        if 'THROAT_DISCOMFORT' in dummy_data_dict: dummy_data_dict['THROAT_DISCOMFORT'] = [1]
        if 'OXYGEN_SATURATION' in dummy_data_dict: dummy_data_dict['OXYGEN_SATURATION'] = [85]
        if 'CHEST_TIGHTNESS' in dummy_data_dict: dummy_data_dict['CHEST_TIGHTNESS'] = [1]
        if 'FATIGUE' in dummy_data_dict: dummy_data_dict['FATIGUE'] = [1]
        if 'COUGHING' in dummy_data_dict: dummy_data_dict['COUGHING'] = [1]
        if 'WHEEZING' in dummy_data_dict: dummy_data_dict['WHEEZING'] = [1]
        if 'SWALLOWING_DIFFICULTY' in dummy_data_dict: dummy_data_dict['SWALLOWING_DIFFICULTY'] = [0]
        if 'ALLERGY' in dummy_data_dict: dummy_data_dict['ALLERGY'] = [1]
        if 'DRY_COUGH' in dummy_data_dict: dummy_data_dict['DRY_COUGH'] = [1]
        if 'FAMILY_HISTORY' in dummy_data_dict: dummy_data_dict['FAMILY_HISTORY'] = [0]
        if 'SMOKING_FAMILY_HISTORY' in dummy_data_dict: dummy_data_dict['SMOKING_FAMILY_HISTORY'] = [1]
        if 'STRESS_IMMUNE' in dummy_data_dict: dummy_data_dict['STRESS_IMMUNE'] = [0]

        # Create DataFrame ensuring correct column order as per feature_names_expected
        dummy_data = pd.DataFrame(dummy_data_dict, columns=feature_names_expected)

        print(f"
Making a prediction for a dummy data point with columns: {dummy_data.columns.tolist()}")
        prediction = make_prediction(loaded_model, dummy_data)
        print(f"Predicted class: {prediction[0]}")

    except FileNotFoundError:
        print("Error: model_1.pkl not found. Please ensure the model is saved in the 'models/' directory.")
    except Exception as e:
        print(f"An error occurred: {e}")
