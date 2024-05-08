# 0차원 스칼라, 1차원 벡터, 2차원 행렬, 3차원 이상 텐서

# 2D Tensor: |t| = (batch size, input dim); (행,렬)
# 3D Tensor(CV): |t| = (batch size, width, height); (행, 렬, 깊이)
# 3D Tensor(NLP): |t| = (batch size, length, input dim); (배치, 단어 수(문장 길이), 단어 벡터의 차원)


import numpy as np

lst = [0, 1, 2, 3, 4, 5, 6]
array = np.array(lst)

Rank = array.ndim
Shape = array.shape

import torch

# Pytorch Tensor Allocation
t = torch.FloatTensor([[1., 2., 3.],
                       [4., 5., 6.],
                       [7., 8., 9.],
                       [10., 11., 12.]])

# Shape, Rank, Axis
dim = t.dim()  # rank, 배열의 차원 수
shape1 = t.shape  # 배열의 크기
shape2 = t.size()  # 배열의 크기

A = torch.FloatTensor([[1, 2], [3, 4]])
B = torch.FloatTensor([[5, 6], [7, 8]])

# Mul vs. Matmul (Broadcasting)
A.mul(B)  # A 텐서와 B 텐서의 원소별 곱셈 연산, 브로드캐스팅 기능 제공
A.matmul(B)  # A 행렬과 B 행렬의 행렬곱, 이때 A의 막 차원과 B의 첫 차원이 같아야 함

# Mean
t.mean()  # 텐서의 평균
t.mean(dim=0)  # 해당 차원(dim)을 제거하여 평균

# Sum
t.sum()  # 텐서의 합계
t.sum(dim=0)  # 해당 차원(dim)을 제거하여 합계

# Max and Argmax
t.max()  # 원소의 최대값(max) 리턴
t.max(dim=0)  # 해당 차원(dim)을 제거하여 (원소의 최댓값(max), 최댓값을 가진 인덱스(argmax)) 리턴

# View
t.view(2, 6)  # 텐서 안의 원소의 개수는 유지하되 텐서의 크기를 변경
t.view(-1, 6)
t.view(3, 2, 2)

# Squeeze
t.squeeze()  # 차원이 1인 차원을 제거
t.squeeze(dim=1)  # 해당 차원(dim)이 1인 경우 해당 차원 제거

# Unsqueeze
t.unsqueeze(dim=1)  # 특정 위치(dim)에 1인 차원을 추가

# Scatter (for one-hot encoding)
lt = torch.LongTensor([[0], [1], [2], [0]])
lt = lt.view(1,4)
one_hot = torch.zeros(3, 4)
one_hot.scatter_(0, lt, 4)  # 인덱싱 축에 대해, 기준 텐서를 원핫코딩으로 바꾸고, 그 값을 채워넣음

# Type Casting
t.long()  # long 타입으로 변환
t.float()  # float 타입으로 변환
t.bool()  # bool 타입으로 변환
t.byte()  # byte 타입으로 변환

# Concatenation
torch.cat([A, B])  # A, B, … 텐서를 모두 연결 (default는 dim=0)
torch.cat([A, B], dim=1)  # A, B, … 텐서를 연결하되 해당 차원(dim)을 늘림

# Stacking
torch.stack([A, B])  # A, B, … 텐서를 모두 쌓아 연결
torch.stack([A, B], dim=1)  # A, B, … 텐서를 모두 쌓되 해당 차원(dim)을 늘림

# Ones and Zeros Like
torch.ones_like(t)  # x 텐서와 동일한 크기지만 1로만 값이 채워진 텐서 생성
torch.zeros_like(t)  # x 텐서와 동일한 크기지만 0으로만 값이 채워진 텐서 생성

# In-place Operation
t.mul_(2.)  # 연산 후 덮어쓰기 수행

# Miscellaneous - Zip
for x, y, z in zip([1, 2, 3], [4, 5, 6], [7, 8, 9]):
    print(x, y, z)
