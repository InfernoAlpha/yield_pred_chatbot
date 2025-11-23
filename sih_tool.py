import joblib
import torch
import pandas as pd
from linear_reg_model import linear_reg

encoder = joblib.load("onehotencoder.pkl")
feature_scaler = joblib.load("minmaxscale1.pkl")
target_scaler = joblib.load("minmaxscale2.pkl")

model = linear_reg(25, 1)
model.load_state_dict(torch.load("model1.pth"))
model.eval()

def predict_crop_yield(user_input: dict) -> float:
    """Predict yield (tons/hectare) for given farm conditions"""
    
    df = pd.DataFrame([user_input])
    
    cat_cols = ['Region', 'Soil_Type', 'Crop', 
                'Weather_Condition', 'Fertilizer_Used', 'Irrigation_Used']
    num_cols = ["Rainfall_mm", "Temperature_Celsius"]

    enc_vals = encoder.transform(df[cat_cols])
    enc_df = pd.DataFrame(enc_vals, columns=encoder.get_feature_names_out(cat_cols))

    full_df = pd.concat([enc_df, df[num_cols]], axis=1)

    scaled = feature_scaler.transform(full_df)
    X = torch.tensor(scaled.astype("float32"))

    with torch.no_grad():
        pred = model(X).numpy()

    pred_real = target_scaler.inverse_transform(pred)[0][0]
    return float(pred_real)