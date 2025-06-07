import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro, levene, mannwhitneyu, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (matplotlib용)
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# -------------------------------
# 1. 데이터 로드 (인코딩 자동 처리)
# -------------------------------
file_path = r"C:\Users\82109\Desktop\고려대\2025-1\한국어정보처리\상반된12개지문추가분석.csv"

encodings_to_try = ['utf-8', 'cp949', 'euc-kr', 'utf-8-sig', 'latin-1']
df = pd.read_csv(file_path, encoding='cp949')

# -------------------------------
# 2. 데이터 전처리
# -------------------------------
# KReaD 이진 분류
def classify_kread_binary(score):
    if pd.isna(score):
        return 'Unknown'
    try:
        score = float(score)
        return 'Easy' if score <= 1799 else 'Hard'
    except:
        return 'Unknown'

df['Kread_binary'] = df['Kread'].apply(classify_kread_binary)

# 표지어 개수를 숫자로 변환
marker_column = '표지어_총합'
df[marker_column] = pd.to_numeric(df[marker_column], errors='coerce')
df_clean = df.dropna(subset=[marker_column])

print(f"전체 샘플 수: {len(df_clean)}")
print(f"사용된 표지어 컬럼: {marker_column}")

# -------------------------------
# 3. 기본 통계량 계산
# -------------------------------
print(f"=== 기본 통계량 ===")

# KReaD 기준
kread_easy = df_clean[df_clean['Kread_binary'] == 'Easy'][marker_column]
kread_hard = df_clean[df_clean['Kread_binary'] == 'Hard'][marker_column]

print(f"KReaD 기준:")
print(f"  Easy: n={len(kread_easy)}, 평균={kread_easy.mean():.2f}, 표준편차={kread_easy.std():.2f}")
print(f"  Hard: n={len(kread_hard)}, 평균={kread_hard.mean():.2f}, 표준편차={kread_hard.std():.2f}")

# 유사도 기준  
sim_easy = df_clean[df_clean['difficulty_median'] == 'Easy'][marker_column]
sim_hard = df_clean[df_clean['difficulty_median'] == 'Hard'][marker_column]

print(f"유사도 기준:")
print(f"  Easy: n={len(sim_easy)}, 평균={sim_easy.mean():.2f}, 표준편차={sim_easy.std():.2f}")
print(f"  Hard: n={len(sim_hard)}, 평균={sim_hard.mean():.2f}, 표준편차={sim_hard.std():.2f}")

# -------------------------------
# 4. 정규성 검정 (Shapiro-Wilk test)
# -------------------------------
print(f"=== 정규성 검정 (Shapiro-Wilk test) ===")

def test_normality(data, name):
    if len(data) < 3:
        print(f"  {name}: 샘플 수 부족 (n={len(data)})")
        return False
    stat, p_value = shapiro(data)
    is_normal = p_value > 0.05
    print(f"  {name}: 통계량={stat:.4f}, p-value={p_value:.4f}, 정규분포={'예' if is_normal else '아니오'}")
    return is_normal

print(f"KReaD 기준:")
kread_easy_normal = test_normality(kread_easy, "Easy")
kread_hard_normal = test_normality(kread_hard, "Hard")

print(f"유사도 기준:")
sim_easy_normal = test_normality(sim_easy, "Easy")
sim_hard_normal = test_normality(sim_hard, "Hard")

# -------------------------------
# 5. 등분산성 검정 (Levene's test)
# -------------------------------
print(f"=== 등분산성 검정 (Levene's test) ===")

def test_homoscedasticity(data1, data2, name):
    if len(data1) < 2 or len(data2) < 2:
        print(f"  {name}: 샘플 수 부족")
        return False
    stat, p_value = levene(data1, data2)
    is_homoscedastic = p_value > 0.05
    print(f"  {name}: 통계량={stat:.4f}, p-value={p_value:.4f}, 등분산성={'예' if is_homoscedastic else '아니오'}")
    return is_homoscedastic

kread_homoscedastic = test_homoscedasticity(kread_easy, kread_hard, "KReaD 기준")
sim_homoscedastic = test_homoscedasticity(sim_easy, sim_hard, "유사도 기준")

# -------------------------------
# 6. t-test 실시
# -------------------------------
print(f"=== t-test 결과 ===")

def perform_ttest(data1, data2, name, assume_equal_var=True):
    if len(data1) < 2 or len(data2) < 2:
        print(f"  {name}: 샘플 수 부족으로 t-test 불가")
        return None, None
    
    # Independent samples t-test
    stat, p_value = stats.ttest_ind(data1, data2, equal_var=assume_equal_var)
    
    # Effect size (Cohen's d) 계산
    pooled_std = np.sqrt(((len(data1)-1)*data1.var() + (len(data2)-1)*data2.var()) / (len(data1)+len(data2)-2))
    cohens_d = (data1.mean() - data2.mean()) / pooled_std if pooled_std != 0 else 0
    
    print(f"{name}:")
    print(f"  t-통계량: {stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Cohen's d (효과크기): {cohens_d:.4f}")
    print(f"  유의성: {'유의함' if p_value < 0.05 else '유의하지 않음'} (α=0.05)")
    
    # Effect size 해석
    if abs(cohens_d) < 0.2:
        effect_size = "작음"
    elif abs(cohens_d) < 0.5:
        effect_size = "중간"
    elif abs(cohens_d) < 0.8:
        effect_size = "큼"
    else:
        effect_size = "매우 큼"
    print(f"  효과크기 해석: {effect_size}")
    
    return stat, p_value

# KReaD 기준 t-test
kread_t_stat, kread_p_val = perform_ttest(kread_easy, kread_hard, "KReaD 기준 (Easy vs Hard)", kread_homoscedastic)

# 유사도 기준 t-test
sim_t_stat, sim_p_val = perform_ttest(sim_easy, sim_hard, "유사도 기준 (Easy vs Hard)", sim_homoscedastic)

# -------------------------------
# 7. 비모수 검정 (Mann-Whitney U test)
# -------------------------------
print(f"=== 비모수 검정 (Mann-Whitney U test) ===")

def perform_mannwhitney(data1, data2, name):
    if len(data1) < 2 or len(data2) < 2:
        print(f"  {name}: 샘플 수 부족으로 Mann-Whitney U test 불가")
        return None, None
    
    stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
    
    print(f"{name}:")
    print(f"  U-통계량: {stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  유의성: {'유의함' if p_value < 0.05 else '유의하지 않음'} (α=0.05)")
    
    return stat, p_value

# 비모수 검정 실시
kread_u_stat, kread_u_p = perform_mannwhitney(kread_easy, kread_hard, "KReaD 기준")
sim_u_stat, sim_u_p = perform_mannwhitney(sim_easy, sim_hard, "유사도 기준")

# -------------------------------
# 8. 분류 일치도 검정 (Chi-square test)
# -------------------------------
print(f"=== 분류 일치도 검정 (Chi-square test) ===")


# -------------------------------
# 10. 시각화
# -------------------------------
print(f"=== 시각화 생성 중... ===")

# 그래프 생성
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('표지어 개수 분포 및 비교 분석', fontsize=16)

# 1. KReaD 기준 박스플롯
kread_data = [kread_easy.tolist(), kread_hard.tolist()]
axes[0,0].boxplot(kread_data, labels=['Easy', 'Hard'])
axes[0,0].set_title('KReaD 기준 표지어 개수 분포')
axes[0,0].set_ylabel('표지어 개수')

# 2. 유사도 기준 박스플롯
sim_data = [sim_easy.tolist(), sim_hard.tolist()]
axes[0,1].boxplot(sim_data, labels=['Easy', 'Hard'])
axes[0,1].set_title('유사도 기준 표지어 개수 분포')
axes[0,1].set_ylabel('표지어 개수')

# 3. KReaD 기준 히스토그램
axes[1,0].hist([kread_easy, kread_hard], label=['Easy', 'Hard'], alpha=0.7, bins=5)
axes[1,0].set_title('KReaD 기준 히스토그램')
axes[1,0].set_xlabel('표지어 개수')
axes[1,0].set_ylabel('빈도')
axes[1,0].legend()

# 4. 유사도 기준 히스토그램
axes[1,1].hist([sim_easy, sim_hard], label=['Easy', 'Hard'], alpha=0.7, bins=5)
axes[1,1].set_title('유사도 기준 히스토그램')
axes[1,1].set_xlabel('표지어 개수')
axes[1,1].set_ylabel('빈도')
axes[1,1].legend()

plt.tight_layout()
plt.savefig('statistical_analysis_results.png', dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------
# 11. 결과 요약 및 저장
# -------------------------------
results_summary = {
    '분석항목': [
        '전체_샘플수', 'KReaD_Easy_평균', 'KReaD_Hard_평균', 'KReaD_t검정_p값',
        '유사도_Easy_평균', '유사도_Hard_평균', '유사도_t검정_p값',
        'KReaD_MannWhitney_p값', '유사도_MannWhitney_p값'
    ],
    '값': [
        len(df_clean), 
        round(kread_easy.mean(), 3) if len(kread_easy) > 0 else 'N/A',
        round(kread_hard.mean(), 3) if len(kread_hard) > 0 else 'N/A',
        round(kread_p_val, 4) if kread_p_val is not None else 'N/A',
        round(sim_easy.mean(), 3) if len(sim_easy) > 0 else 'N/A',
        round(sim_hard.mean(), 3) if len(sim_hard) > 0 else 'N/A',
        round(sim_p_val, 4) if sim_p_val is not None else 'N/A',
        round(kread_u_p, 4) if kread_u_p is not None else 'N/A',
        round(sim_u_p, 4) if sim_u_p is not None else 'N/A'
    ]
}

results_df = pd.DataFrame(results_summary)
results_df.to_csv('statistical_test_results.csv', index=False, encoding='utf-8-sig')



'''
결과
전체 샘플 수: 12
사용된 표지어 컬럼: 표지어_총합
=== 기본 통계량 ===

KReaD 기준:
  Easy: n=7, 평균=9.57, 표준편차=7.14
  Hard: n=5, 평균=7.60, 표준편차=3.36
유사도 기준:
  Easy: n=5, 평균=7.60, 표준편차=3.36
  Hard: n=7, 평균=9.57, 표준편차=7.14
=== 정규성 검정 (Shapiro-Wilk test) ===
KReaD 기준:
  Easy: 통계량=0.7383, p-value=0.0096, 정규분포=아니오
  Hard: 통계량=0.9245, p-value=0.5597, 정규분포=예
유사도 기준:
  Easy: 통계량=0.9245, p-value=0.5597, 정규분포=예
  Hard: 통계량=0.7383, p-value=0.0096, 정규분포=아니오
=== 등분산성 검정 (Levene's test) ===
  KReaD 기준: 통계량=0.4879, p-value=0.5008, 등분산성=예
  유사도 기준: 통계량=0.4879, p-value=0.5008, 등분산성=예
=== t-test 결과 ===
KReaD 기준 (Easy vs Hard):
  t-통계량: 0.5684
  p-value: 0.5823
  Cohen's d (효과크기): 0.3328
  유의성: 유의하지 않음 (α=0.05)
  효과크기 해석: 중간
유사도 기준 (Easy vs Hard):
  t-통계량: -0.5684
  p-value: 0.5823
  Cohen's d (효과크기): -0.3328
  유의성: 유의하지 않음 (α=0.05)
  효과크기 해석: 중간
=== 비모수 검정 (Mann-Whitney U test) ===
KReaD 기준:
  U-통계량: 19.5000
  p-value: 0.8065
  유의성: 유의하지 않음 (α=0.05)
유사도 기준:
  U-통계량: 15.5000
  p-value: 0.8065
  유의성: 유의하지 않음 (α=0.05)
=== 분류 일치도 검정 (Chi-square test) ===
=== 시각화 생성 중... ===
'''
