import torch 
import torch.nn as nn
import joblib
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from torch.utils.data import random_split
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import r2_score
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import io
import seaborn as sns
from sklearn.metrics import confusion_matrix
from PIL import Image

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
scale = MinMaxScaler()
target = MinMaxScaler()

class csv_dataset(Dataset):
    def __init__(self,csv_filepath):
        super().__init__()
        df = pd.read_csv(csv_filepath)
        
        colu = ['Region', 'Soil_Type', 'Crop', 'Weather_Condition','Fertilizer_Used','Irrigation_Used']
        numeric_cols = ["Rainfall_mm", "Temperature_Celsius"]
        enc_colu = encoder.fit_transform(df[colu])
        enc_df = pd.DataFrame(enc_colu,columns=encoder.get_feature_names_out(colu))

        feature_df = pd.concat([enc_df, df[numeric_cols]], axis=1)
        target_df = pd.DataFrame(df['Yield_tons_per_hectare'])

        norm_features = scale.fit_transform(feature_df)
        norm_target = target.fit_transform(target_df)

        joblib.dump(encoder,"onehotencoder.pkl")
        joblib.dump(scale,"minmaxscale1.pkl")
        joblib.dump(target,"minmaxscale2.pkl")

        self.X = torch.tensor(norm_features.astype(np.float32))
        self.y = torch.tensor(norm_target.astype(np.float32))

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index],self.y[index]

class linear_reg(nn.Module):
    def __init__(self,input,output):
        super().__init__()

        self.linear1 = nn.Linear(input, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, output)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

        self.l_relu = nn.LeakyReLU()

    def forward(self,x):
        x = self.l_relu(self.bn1(self.linear1(x)))
        x = self.l_relu(self.bn2(self.linear2(x)))
        x = self.l_relu(self.bn3(self.linear3(x)))
        x = self.linear4(x)
        return x

def R2_score(model,loader,device = "cuda" if torch.cuda.is_available() else "cpu"):
    model = model.to(device)
    model.eval()
    all_targets = []
    all_preds = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            output = model(x)
            all_preds.append(output.view(-1).cpu())
            all_targets.append(y.view(-1).cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    score = r2_score(all_targets, all_preds)
    return score

def log_regression_plot(model, loader, writer, epoch, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            output = model(x)
            all_preds.append(output.view(-1).cpu())
            all_targets.append(y.view(-1).cpu())
    
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    figure = plt.figure(figsize=(8, 8))
    plt.scatter(targets, preds, alpha=0.5, color='blue')
    
    limit_min = min(targets.min(), preds.min())
    limit_max = max(targets.max(), preds.max())
    plt.plot([limit_min, limit_max], [limit_min, limit_max], 'r--', label='Perfect Prediction')
    
    plt.xlabel('Actual Yield (Normalized)')
    plt.ylabel('Predicted Yield (Normalized)')
    plt.title(f'Actual vs Predicted (Epoch {epoch})')
    plt.legend()
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = Image.open(buf)
    image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1)

    writer.add_image("Evaluation/Regression_Plot", image_tensor, epoch)

def train(model,train_dataloader,epochs=20,lr=0.001,device = "cuda" if torch.cuda.is_available() else "cpu"):
    model = model.to(device)
    loss_fn = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(params=model.parameters(),lr=lr)
    writer = SummaryWriter(log_dir="runs/crop_yield_predictor_run_1")
    batch = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for x,y in train_dataloader:
            x,y = x.to(device),y.squeeze().to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y.unsqueeze(1))
            loss.backward()
            optimizer.step()

            writer.add_scalar("Loss/Batch",loss.item(),batch)
            batch += 1
            epoch_loss += loss.item()
        log_regression_plot(model, test_dataloader, writer, epoch, device)
        print(f"epoch:{epoch},loss:{loss}")
        avg_loss = epoch_loss / len(train_dataloader)
        writer.add_scalar("Loss/Epoch",avg_loss,epoch)
        test_score = R2_score(model, test_dataloader, device)
        writer.add_scalar("R2_score/Test", test_score, epoch)
    writer.flush()
    writer.close()

    torch.save(model.state_dict(),"model1.pth")
    return model

if __name__ == "__main__":
    data = csv_dataset(r"C:\Users\chara\OneDrive\Desktop\Desktop\vs code\langchain\sih\crop_yield.csv")
    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size
    train_data, test_data = random_split(data, [train_size, test_size])

    train_dataloader = DataLoader(dataset=train_data,shuffle=True,batch_size=128)
    test_dataloader = DataLoader(dataset=test_data,shuffle=True,batch_size=128)
    model = linear_reg(25,1)
    #model = train(model,train_dataloader)
    #print(R2_score(model=model,loader=test_dataloader))
    model.load_state_dict(torch.load("model1.pth",weights_only=True))
    encoder = joblib.load("onehotencoder.pkl")
    scale = joblib.load("minmaxscale1.pkl")
    target = joblib.load("minmaxscale2.pkl")

    raw_data = {
        'Region': ['West'],
        'Soil_Type': ['Sandy'],
        'Crop': ['Cotton'],
        'Weather_Condition': ['Cloudy'],
        'Fertilizer_Used': [False], 
        'Irrigation_Used': [True],
        'Rainfall_mm': [897.0772391101236],
        'Temperature_Celsius': [27.676966373377603]
    }
    
    df = pd.DataFrame(raw_data)
    colu = ['Region', 'Soil_Type', 'Crop', 'Weather_Condition','Fertilizer_Used','Irrigation_Used']
    numeric_cols = ["Rainfall_mm", "Temperature_Celsius"]
    enc_colu = encoder.transform(df[colu])
    enc_df = pd.DataFrame(enc_colu,columns=encoder.get_feature_names_out(colu))

    feature_df = pd.concat([enc_df, df[numeric_cols]], axis=1)

    norm_features = scale.transform(feature_df)

    X = torch.tensor(norm_features.astype(np.float32))
    model.eval()
    y = model(X)

    print(f"predicted yield:{target.inverse_transform(y.cpu().detach().numpy())[0][0]}")