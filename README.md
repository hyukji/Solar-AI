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
    -> 1.93
+ **20210103_2305.csv**: 4days include hour
+ **20210103_2340.csv**: catboost test
+ **20210104_0012.csv**: catboost 4days
    -> DHI가 0인 날 예측값 0.01이 많이 나와서 0인날 제거해 볼 예정
+ **20210104_1408.csv**: catboost 4days 0제거 10000
+ **20210104_1612.csv**: catboost 4days 0제거, n_estimators 1000
    -> 결과가 안좋게나와서 그냥 LGBM feature랑 하이퍼파라미터를 최적화하는게 더 좋을듯
+ **20210104_1621.csv**: LGBM 4days estimator 2000
+ **20210104_1730.csv**: = 10000
+ **20210104_1859.csv**: LGBM 0제거
+ **20210105_1340.csv**: kfold units 4, 2000 LGBM
+ **20210105_1350.csv**: kfold units 4, 4000 LGBM
    -> 1.94점