import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score

def train_model(model, criterion, optimizer, x_train, y_train, x_test, y_test, batch_size, epochs, patience):
    train_losses = []
    val_losses = []
    val_r2_scores = []  # R² 스코어 저장 리스트

    # Early stopping 설정
    best_val_loss = float('inf')
    patience_counter = 0

    # 데이터 로더 생성
    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).view(-1, 1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print(train_dataset)
    print(train_loader)

    for epoch in range(epochs):
        model.train()  # 학습 모드 활성화
        epoch_loss = 0

        for batch_X, batch_y in train_loader:
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))

        # 검증 손실 및 R² 스코어 계산
        model.eval()
        with torch.no_grad():
            y_pred_val = model(torch.tensor(x_test, dtype=torch.float32))
            val_loss = criterion(y_pred_val, torch.tensor(y_test, dtype=torch.float32).view(-1, 1))
            val_losses.append(val_loss.item())

            # R² 스코어 계산
            r2 = r2_score(y_test, y_pred_val.numpy())
            val_r2_scores.append(r2)

        # Early stopping 체크
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"학습 조기 종료 {epoch + 1}")
            break

        # Epoch 결과 출력
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], "
                  f"훈련 Loss: {train_losses[-1]:.4f}, 검증 Loss: {val_losses[-1]:.4f}, "
                  f"검증 R²: {val_r2_scores[-1]:.4f}")

    return train_losses, val_losses, val_r2_scores
