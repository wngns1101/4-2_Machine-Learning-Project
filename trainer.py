import pandas as pd
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim import Adam
import utils as util
import linearModel
import train
import random
import numpy as np
import torch

# 시드 고정
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 데이터 로드
data = pd.read_csv("./data/final_data.csv")

x, y = util.preprocess_data(data)

# 데이터 로드 및 전처리
x_scaled, y_scaled, scaler_x, scaler_y = util.standardize_data(x, y)

# 학습 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=42)

# 모델 정의
input_dim = x_train.shape[1]
model = linearModel.LinearModel(input_dim)

# 손실 함수 및 옵티마이저 정의
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.001)

train_losses, val_losses, val_r2_scores = train.train_model(
    model, criterion, optimizer, x_train, y_train, x_test, y_test, 128, 100, 30
)

# 최종 결과 출력
print(f"마지막 훈련 Loss 값: {train_losses[-1]:.4f}")
print(f"마지막 검증 Loss 값: {val_losses[-1]:.4f}")
print(f"마지막 R² Score 값: {val_r2_scores[-1]:.4f}")

# 손실 시각화
util.plot_losses(train_losses, val_losses)

# 모델 저장
torch.save(model.state_dict(), "model/model.pth")
