from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    # 필터링 및 열 제거
    data = data[data['position_encoded'] == 4]
    data = data.drop(columns=["player", "position", "team", "name", "goals conceded", "clean sheets", "winger"])

    # 독립 변수(X)와 종속 변수(y) 분리
    x = data.drop(columns=["current_value"]).values
    y = data["current_value"].values

    return x, y


def standardize_data(x, y):
    # 데이터 정규화
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    x_scaled = scaler_x.fit_transform(x)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    # scaler_x와 scaler_y를 반환
    return x_scaled, y_scaled, scaler_x, scaler_y

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.legend()
    plt.show()


def plot_prediction_result(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, label="Predictions", color="blue")
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color="red", linewidth=2, label="Perfect Fit")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("True vs Predicted Values")
    plt.legend()
    plt.show()