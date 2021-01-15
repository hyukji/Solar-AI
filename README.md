# Solar-AI

## 폴더 및 파일 섦명
- data : train, test, sample_submission 데이터
- base_code : baseline 코드 그대로
- submission : 제출용 csv 파일들
- module : 필요한 함수들
	- data.py
		- 데이터 전처리를 위한 함수 포함, 연속된 데이터 개수(cons)와 시간 간격(unit), 제거할 열(removed_column)을 지정함.
		- 동시간대가 아닌 데이터도 학습시키려고 함. (ex. cons=2, unit=1 => 현재와 직전 시간의 데이터를 기반으로)
	- lgbm.py : lgbm 모델
	- deep.py : keras.dense 모델

- lgbm.py : lgbm.py의 메인 파일
- deep.py : deep.py의 메인 파일
- season.py : lgbm.py 기반, TARGET을 기준으로 3계절을 구분해서 학습시킴. (코드 정리 안 됨)
- around.py : lgbm.py 기반, 연속된 날짜의 특정 시간대의 데이터를 학습시키기 위한 코드
- comparison.py : 다양한 모델을 학습시키고 비교하는 파일. (미완성)
- practice.py : code sandbox 및 메모

## 활용한 모델
1. LGBM 
	- (main.py) 동시간대의 연속적인 나흘로 학습했을 때 1.93의 결과가 나옴.(ie. cons=4, unit=48) 닷새, 엿새는 제출해보지 않음. 
	- (module/lgbm.py) lightgbm.plot_importance 함수로 feature의 중요도를 확인했는데 중요도가 낮은 column(Day, T etc)은 지워도 별 영향이 없음.
2. keras NN (DEEP.py): feature를 많이 넣어야 하고 학습 시간이 오래 걸림. validation loss는 낮은데 제출하면 2.2에 머묾.
3. keras RNN LSTM (RNN.py) : 특별한 이점이 없음
4. OLS, Quantile Regreesion, Random Forests, Gradient Boosting (comparison.py)
	- https://colab.research.google.com/drive/1nXOlrmVHqCHiixqiMF6H8LSciz583_W2#scrollTo=f4nQmq2b9H7s

## 다른 시도
1. (season.py) TARGET을 기준으로 계절을 winter, fall(=spring), summer로 나누어 학습했는데 2.23의 결과가 나옴. 많이 시도해보지 않음.