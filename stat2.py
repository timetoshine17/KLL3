import pandas as pd
import numpy as np

# 1) 예시 데이터 (이미 rows 리스트가 있다면 이 부분은 생략하세요)
rows = [
    {'file': '20210602', 'mean_sim': 2.200950},
    {'file': '20210603', 'mean_sim': 1.443633},
    {'file': '20210902', 'mean_sim': 2.384800},
    {'file': '20210903', 'mean_sim': 1.505133},
    {'file': '20211102', 'mean_sim': 1.481400},
    {'file': '20211103', 'mean_sim': 2.111250},
    {'file': '20220602', 'mean_sim': 0.667633},
    {'file': '20220603', 'mean_sim': 1.169142},
    {'file': '20220901', 'mean_sim': 1.485283},
    {'file': '20220903', 'mean_sim': 1.083617},
    {'file': '20221102', 'mean_sim': 2.212250},
    {'file': '20221103', 'mean_sim': 2.284400},
    {'file': '20230602', 'mean_sim': 1.431183},
    {'file': '20230603', 'mean_sim': 1.479817},
    {'file': '20230902', 'mean_sim': 1.594433},
    {'file': '20230903', 'mean_sim': 1.446000},
    {'file': '20231102', 'mean_sim': 2.064800},
    {'file': '20231103', 'mean_sim': 0.792655},
]

# 2) DataFrame 생성
df = pd.DataFrame(rows)

# 3) inf, nan 처리 (inf를 제외하거나 nan으로 바꿀지 선택)
#    여기서는 inf를 제외한 뒤 계산합니다.
df_valid = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['mean_sim'])

# 4) 통계값 계산
median = df_valid['mean_sim'].median()
q1     = df_valid['mean_sim'].quantile(0.25)
q2     = df_valid['mean_sim'].quantile(0.50)  # 중앙값과 동일
q3     = df_valid['mean_sim'].quantile(0.75)
mean   = df_valid['mean_sim'].mean()

# 5) 결과 출력
print(f"평균(mean)       : {mean:.6f}")
print(f"중앙값(median)   : {median:.6f}")
print(f"1사분위(Q1, 25%) : {q1:.6f}")
print(f"2사분위(Q2, 50%) : {q2:.6f}")
print(f"3사분위(Q3, 75%) : {q3:.6f}")
