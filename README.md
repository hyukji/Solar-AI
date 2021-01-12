Solar-AI
===
파일
---
+ **baseline.py** 베이스라인코드 
+ **fixedBaseline.py** 베이스라인 코드에서 쓸모없는 부분 제거, 수정
+ **main.py** LGBM 
+ **cat_main.py** CatBoost maim
+ **kfold_main.py** kfold를 이용해 학습
+ **analysis_data.py** 데이터 분석용
### moudules
+ **data_process.py** 데이터 불러오기, 전처리
+ **LGBM.py** lightgbm 모델 훈련, 예측
+ **CATBM.py** CatBoost 모델 훈련, 예측
+ **load_mvag.py** 이동평균 불러오기
+ **load_change.py** 변화량 불러오기

발전량예측과 기상예측 분리 ,, ?
시간대별로도 변화량
lgbm lr decay

논문 참고
1. 발전량, 기상 분리
2. 이동평균, 기울기
3. GHI