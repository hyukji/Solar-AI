# 최적 파라미터 찾기
# 딥러닝
# 퀀타일 분포를 넓게 바꿔보기
# 전날이 아닌 6일치의 데이터를 활용하기
# 여러 모델의 배깅(선형 결합)
# 유의미한 변수 생성
# -(DHI+DNI)
# -몇월인지 구분해서 각기 다른 모델로? (버스 시간에서 주말/평일을 구분한 것처럼)
# 무의미한 변수 삭제
# 지금은 같은 시각의 데이터로만 다음날을 예측함.
# -오후 23시의 날씨가 내일 오후3시에 영향을 줄 수도 있음.
# -feature가 많아지니 딥러닝 쓸 수 있을 듯?

import pandas as pd
import numpy as np
preds = pd.Series([np.nan]*4)
season = pd.Series(['winter', 'fall', 'summer', 'summer'])
for i, v in enumerate(['winter', 'fall', 'summer']):
    pred = pd.Series(range(i, i+4)).where(season == v)
    preds = preds.where(preds>=0, pred)
    print(preds)

days = 4
for day in range(6, 6-days, -1):
    print(day)



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