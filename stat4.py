import pandas as pd
import numpy as np

# 1) 데이터 정의
rows = [
    {'file': '20210601', 'mean_sim': 0.724724},
    {'file': '20210901', 'mean_sim': 0.887473},
    {'file': '20211101', 'mean_sim': 0.668819},
    {'file': '20220601', 'mean_sim': 0.651538},
    {'file': '20220902', 'mean_sim': 0.818530},
    {'file': '20221101', 'mean_sim': 0.805487},
    {'file': '20230601', 'mean_sim': 0.659377},
    {'file': '20230901', 'mean_sim': 1.005613},
    {'file': '20231101', 'mean_sim': 0.854760},
]

df = pd.DataFrame(rows)

# 2) 통계값 계산
mean_val   = df['mean_sim'].mean()
median_val = df['mean_sim'].median()
q1         = df['mean_sim'].quantile(0.25)
q3         = df['mean_sim'].quantile(0.75)

print(f"Mean   : {mean_val:.6f}")
print(f"Median : {median_val:.6f}")
print(f"Q1 (25%): {q1:.6f}")
print(f"Q3 (75%): {q3:.6f}\n")

# 3) 중앙값 기준 분류 (2단계)
df['difficulty_median'] = np.where(
    df['mean_sim'] >= median_val,
    'Easy',
    'Hard'
)

# 4) 사분위 기준 분류 (3단계)
def quartile_category(x):
    if x > q3:
        return 'Easy'
    elif x < q1:
        return 'Hard'
    else:
        return 'Medium'

df['difficulty_quartile'] = df['mean_sim'].apply(quartile_category)

# 5) 결과 출력
print(df.to_string(index=False))
