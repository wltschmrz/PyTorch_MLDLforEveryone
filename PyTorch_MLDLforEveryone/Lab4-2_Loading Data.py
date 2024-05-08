# • MiniBatch: 전체 데이터를 더 작은 단위로 나눈 해당 단위
# • BatchGradientdescent: '전체' 하강법을 수행하는 방법
# • MinibatchGradientdescent: '미니 배치' 단위로 경사하강법을 수행하는 방법

# 업데이트를 빠르게 가능, 표본의 크기가 적어지기에 잘못된 방향으로 업데이트 할 수도 있다.
# 배치 크기는 보통 2의 제곱수를 사용 - CPU와 GPU의 메모리가 2의 배수라서 데이터 송수신의 효율을 높이기 위해

# • Epoch: 전체 훈련 데이터가 학습에 한 번 사용된 주기
# • Iteration: 한 번의 Epoch 내에서 이뤄지는 매개변수인 가중치의 업데이트 횟수

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset  # 텐서데이터셋
from torch.utils.data import DataLoader  # 데이터로더

x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

dataset = TensorDataset(x_train, y_train)

dataloader = DataLoader(dataset, batch_size=2, shuffle=True)  # shuffle은 epoch마다 dataset을 섞어줌 - 문제 순서 암기 방지

model = nn.Linear(3, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20
for epoch in range(1, nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader):
        x_train_sample, y_train_sample = samples

        # H(x) 계산
        prediction = model(x_train_sample)

        # cost 계산
        cost = F.mse_loss(prediction, y_train_sample)

        # cost로 H(x) 계산
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, batch_idx + 1, len(dataloader),
            cost.item()
        ))
# ____________________________________________________________________________________________________
# Dataset을 커스텀 할 수 있다.

from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, x, y):  # 데이터셋의 전처리 해주는 부분
        super().__init__()
        self.x_data = x
        self.y_data = y

    def __len__(self):  # 데이터셋의 길이, 총 샘플 수를 알려주는 부분
        return len(self.x_data)

    def __getitem__(self, idx):  # 데이터셋에서 특정 1개의 샘플을 가져오는 부분
        return self.x_data[idx], self.y_data[idx]


x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

dataset2 = CustomDataset(x_train, y_train)

dataloader2 = DataLoader(dataset2, batch_size=2, shuffle=True)

model = nn.Linear(3, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20
for epoch in range(1, nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader2):
        x_train_sample, y_train_sample = samples

        # H(x) 계산
        prediction = model(x_train_sample)

        # cost 계산
        cost = F.mse_loss(prediction, y_train_sample)

        # cost로 H(x) 계산
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, batch_idx + 1, len(dataloader2),
            cost.item()
        ))
