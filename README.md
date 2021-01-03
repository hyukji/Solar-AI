Solar-AI
===
파일
---
+ **baseline.py** 베이스라인코드 
+ **fixedBaseline.py** 베이스라인 코드에서 쓸모없는 부분 제거, 수정
+ **main.py** LGBM 
+ **cat_main.py**
### moudules
+ **data_process.py** 데이터 불러오기, 전처리
+ **LGBM.py** lightgbm 모델 훈련, 예측
+ **CATBM.py** CatBoost 모델 훈련, 예측
### submissions files
+ **20210103_2255.csv**: 4days without hour
+ **20210103_2305.csv**: 4days include hour
+ **20210103_2340.csv**: catboost test
+ **20210104_0012.csv**: catboost 4days
    -> DHI가 0인 날 예측값 0.01이 많이 나와서 0인날 제거해 볼 예정
