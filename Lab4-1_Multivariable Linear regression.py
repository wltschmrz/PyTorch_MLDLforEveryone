# • 다중 선형 회귀(Multiple Linear Regression): x가 2개 이상인 선형 회귀(x로부터 y를 예측하는 1차식)
# • 다변량 선형 회귀(Multivariate/Multivariable Linear Regression): y가 2개 이상인 선형 회귀
# • 샘플(sample): 전체 훈련 데이터의 개수를 셀 수 있는 1개의 단위
# • 특성(feature): 각 샘플에서 y를 결정하게 하는 각각의 독립 변수 x

# Hypothesis Function: Naive vs Matrix (Scalar 식으로 쓰는 것보다 Matrix가 속도 및 가시성에 좋다)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
# ____________________________________________________________________________________________________
# Multiple Linear Regression

x_train = torch.FloatTensor([[73, 80, 75],  # 5 x 3
                             [93, 88, 93],
                             [89, 91, 80],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])  # 5 x 1

# 모델 초기화
W = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=1e-5)

nb_epochs = 20
for epoch in range(1, nb_epochs + 1):
    # H(x) 계산
    # 편향 b는 브로드 캐스팅되어 각 샘플에 더해집니다.
    hypothesis = x_train.matmul(W) + b  # 한 행에 한 sample을 담을거라 식의 개형이 'XW + b'가 자연스럽다.

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(
        epoch, nb_epochs, hypothesis.squeeze().detach(), cost.item()
    ))


# ____________________________________________________________________________________________________
# nn.Module & F.mse_loss

class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)  # 입력 차원: 3, 출력 차원: 1

    def forward(self, x):  # Hypothesis 계산하는 곳, Forward pass
        return self.linear(x)


# 모델 초기화
model = MultivariateLinearRegressionModel()
# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20
for epoch in range(1, nb_epochs + 1):
    # H(x) 계산
    hypothesis = model(x_train)

    # cost 계산
    cost = F.mse_loss(hypothesis, y_train)

    # Gradient Descent
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print(f"Epoch {epoch:4d}/{nb_epochs} \
    hypothesis: {hypothesis.squeeze().detach()} \
    Cost: {cost.item():.6f}")
