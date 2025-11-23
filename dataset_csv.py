import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
import numpy as np
from torch.utils.data import Dataset

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
scale = MinMaxScaler()

class csv_dataset(Dataset):
    def __init__(self,csv_filepath):
        super().__init__()
        df = pd.read_csv(csv_filepath)
        
        colu = ['Region', 'Soil_Type', 'Crop', 'Weather_Condition','Fertilizer_Used','Irrigation_Used']
        numeric_cols = ["Rainfall_mm", "Temperature_Celsius", "Days_to_Harvest"]
        enc_colu = encoder.fit_transform(df[colu])
        enc_df = pd.DataFrame(enc_colu,columns=encoder.get_feature_names_out(colu))

        full_df = pd.concat([enc_df,df[numeric_cols],df['Yield_tons_per_hectare']],axis=1)


        norm_data = scale.fit_transform(full_df)
        df = pd.DataFrame(norm_data,columns=full_df.columns)
        self.X = torch.tensor(df.iloc[:, :-1].values.astype(np.float32))
        self.y = torch.tensor(df.iloc[:, -1].values.astype(np.float32))

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index],self.y[index]
a = csv_dataset(r'C:\Users\chara\Desktop\Desktop\vs code\langchain\sih\crop_yield.csv')