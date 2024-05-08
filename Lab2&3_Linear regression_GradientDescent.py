import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_train = torch.FloatTensor([[1], [2], [3]])  # 데이터
y_train = torch.FloatTensor([[2], [4], [6]])

W = torch.zeros(1, requires_grad=True)  # W, b 초기화
b = torch.zeros(1, requires_grad=True)  # requires_grad=True: 학습할 것이라고 명시 *

optimizer = optim.SGD([W, b], lr=0.01)  # optimizer 설정


n_epochs = 2000  # 원하는만큼 경사 하강법을 반복

for epoch in range(1, n_epochs+1):

    # H(x) 계산
    hypothesis = x_train * W + b

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()  # gradient를 0으로 초기화 *
    cost.backward()  # 비용 함수를 미분하여 gradient 계산 *
    optimizer.step()  # W와 b를 업데이트 *

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, n_epochs, W.item(), b.item(), cost.item()
        ))

# ____________________________________________________________________________________________________

# optimizer는 아래 gradient descent 과정을 수행해주는 것

# gradient = 2 * torch.mean((W * x_train - y_train) * x_train)
# lr = 0.1
# W -= lr * gradient
