# Solar-AI

## 폴더 및 파일 섦명
&#10148; data : train, test, sample_submission 데이터

&#10148; base_code : baseline 코드 그대로

&#10148; submission : 제출용 csv 파일들

- module.py : 데이터 전처리를 위한 함수 포함, 연속된 데이터 개수(cons)와 시간 간격(unit), 제거할 열(removed_column)을 지정함.
- main.py : module과 LGBM을 활용한 기본 실행 파일
- deep.py : 저장된 pickle을 불러와 keras.dense 모델을 학습시킴.
- season.py : TARGET을 기준으로 3계절을 구분해서 LGBM을 학습시킴. (코드 정리 안 됨)
- comparison.py : 다양한 모델을 학습시키고 비교하는 파일. (미완성)
- around.py : (미완성)
- practice.py : code sandbox 및 메모

## 진행 상황
1. (main.py) 동시간대의 연속적인 나흘로 학습했을 때 1.93의 결과가 나왔고 닷새, 엿새는 제출해보지 않음.
2. (module.py) unit 단위를 추가해 동시간대가 아닌 데이터도 학습시키려고 함.
3. (main.py) lightgbm.plot_importance 함수로 feature의 중요도를 확인했는데 중요도가 낮은 column(Day, T etc)은 지워도 별 영향이 없음. '토론'의 분석 내용에서 GHI 변수가 좋다고 해서 추가했는데 plot_importance에서 큰 영향이 없었음.
4. (season.py) TARGET을 기준으로 계절을 winter, fall(=spring), summer로 나누어 학습했는데 2.23의 결과가 나옴. 많이 시도해보지 않음.
5. (comparison.py) quantile regression 모델을 비교하는 사이트를 보고 공부하는 중임. https://colab.research.google.com/drive/1nXOlrmVHqCHiixqiMF6H8LSciz583_W2#scrollTo=f4nQmq2b9H7s
6. (deep.py) 데이터 dimesion을 키워서 학습해보는 것이 도움이 될 거 같음. 출력되는 vali_loss가 LGBM과 동일해도 제출 시 성능은 더 낮게 나옴.
