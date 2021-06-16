# https://www.codeit.kr/learn/courses/machine-learning/3345
# Grid Search: 최적의 하이퍼파라미터 고르는 방법

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

import numpy as np
import pandas as pd

# 경고 메시지 출력 억제 코드
import warnings
warnings.simplefilter(action='ignore')

GENDER_FILE_PATH = './datasets/gender.csv'

# 데이터 셋을 가지고 온다
gender_df = pd.read_csv(GENDER_FILE_PATH)

X = pd.get_dummies(gender_df.drop(['Gender'], axis=1)) # 입력 변수를 one-hot encode한다
y = gender_df[['Gender']].values.ravel()

log_model = LogisticRegression()
hyper_param = {
    'penalty': ['l1', 'l2'],
    'max_iter': [500, 1000, 1500, 2000]
}
hyper_param_tuner = GridSearchCV(log_model, hyper_param, cv=5)
hyper_param_tuner.fit(X, y)

best_params = hyper_param_tuner.best_params_

# 체점용 코드
print(best_params)