# feature 추가 및 삭제
# Hour, Minute 제거는 모델에 따라 성능이 올라갈 수도 있고 아닐 수도 있음.
# GHI 칼럼 추가하기 (태양과의 각도 계산이 필요함.)
# 변화량 칼럼 추가
# 이평선을 칼럼에서 추가할 수도 있음.

# 활용한 모델
# LGBM
# keras NN : feature를 많이 넣어야 함.
# keras RNN LSTM : 특별한 이점이 없음

# 추가적인 기법
# KFold croos validation
# lr_scheduler
# early stopping
# modelcheckpoint

# 새로운 시도
# 계절 구분해서 각기 다른 모델 학습시키기

# 시도할 것
# 퀀타일 분포를 바꾸어보기 (토론에서 분포를 넓게 했더니 더 좋아졌다는 말이 있음)
# 밤 데이터를 아예 빼버리고 모델 학습시키기
# 여러 모델의 배깅(선형 결합)
# 최적 파라미터 찾기 (gridsearchsv?)

# import pandas as pd
# import numpy as np
# preds = pd.Series([np.nan]*4)
# season = pd.Series(['winter', 'fall', 'summer', 'summer'])
# for i, v in enumerate(['winter', 'fall', 'summer']):
#     pred = pd.Series(range(i, i+4)).where(season == v)
#     preds = preds.where(preds>=0, pred)
#     print(preds)

# days = 672
# round(days ** (1/3))

import numpy as np
p = np.array([1, 2, 3, 4]).reshape(1, -1)
np.percentile(p, [10, 20, 30, 40, 50, 60, 70, 80, 90])

def tilted_loss(q,y,f):
    e = (y-f)
    return np.mean(np.maximum(q*e, (q-1)*e))

a=[3.97,	6.16,	6.43,	6.52,	6.06,	5.10,	6.95,	6.82,	7.28]
b=[4.874,	5.676,	6.100,	6.214,	6.43,	6.502,	6.700,	6.872,	7.016]
res = []
for i in range(9):
    q = (i+1)*0.1
    r = tilted_loss(q, 8.01, a[i])
    res.append(r)
print(res)
print(round(sum(res)/len(res), 3))



# # Creating the First Series 
# sr1 = pd.Series([22, 18, 19, 20, 21]) 
  
# # Creating the row axis labels 
# sr1.index = ['Student 1', 'Student 2', 'Student 3', 'Student 4', 'Student 5'] 
  
# # Print the series 
# print(sr1) 
  
# # Creating the second Series 
# sr2 = pd.Series([19, 16, 22, 20, 18]) 
  
# # Creating the row axis labels 
# sr2.index = ['Student 1', 'Student 2', 'Student 3', 'Student 4', 'Student 5'] 
  
# # Print the series 
# print(sr2) 
# sr1.where(sr1 >20) 