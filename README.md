Solar-AI
===
����
---
+ **baseline.py** ���̽������ڵ� 
+ **fixedBaseline.py** ���̽����� �ڵ忡�� ������� �κ� ����, ����
+ **main.py** LGBM 
+ **cat_main.py**
### moudules
+ **data_process.py** ������ �ҷ�����, ��ó��
+ **LGBM.py** lightgbm �� �Ʒ�, ����
+ **CATBM.py** CatBoost �� �Ʒ�, ����
### submissions files
+ **20210103_2255.csv**: 4days without hour
    -> 1.93
+ **20210103_2305.csv**: 4days include hour
+ **20210103_2340.csv**: catboost test
+ **20210104_0012.csv**: catboost 4days
    -> DHI�� 0�� �� ������ 0.01�� ���� ���ͼ� 0�γ� ������ �� ����
+ **20210104_1408.csv**: catboost 4days 0���� 10000
+ **20210104_1612.csv**: catboost 4days 0����, n_estimators 1000
    -> ����� �����Գ��ͼ� �׳� LGBM feature�� �������Ķ���͸� ����ȭ�ϴ°� �� ������
+ **20210104_1621.csv**: LGBM 4days estimator 2000
+ **20210104_1730.csv**: = 10000
+ **20210104_1859.csv**: LGBM 0����
+ **20210105_1340.csv**: kfold units 4, 2000 LGBM
+ **20210105_1350.csv**: kfold units 4, 4000 LGBM
    -> 1.94��
+ **20210105_1445.csv**: kfold units 5, 3000 LGBM
+ **20210105_2135.csv**: kfold units 5, 3000 LGBM, ������
    -> 2.12
+ **20210106_1355.csv**: kfold units 4, 1000 catbm, ������
+ **20210106_1405.csv**: kfold units 4, 3000 catbm, ������
+ **20210106_1430.csv**: kfold units 4, 2000 catbm
    -> 1.91
+ **20210106_1823.csv**: kfold units 4, 2000 catbm, ������
+ **20210106_1846.csv**: 1430 + 2255 bagging 1:1
    -> 1.89
+ **20210107_1617.csv**: kfold units 4, 2000 catbm, �̵����
������������ ����� �и� ,, ?

�� ����
1. ������, ��� �и�
2. �̵����, ����
3. GHI