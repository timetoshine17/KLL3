import pandas as pd

# -------------------------------
# 1. 데이터 불러오기
# -------------------------------
file_path = r"C:\Users\82109\Desktop\고려대\2025-1\한국어정보처리\유사도-KReaD_상반_지문_binary.csv"
df = pd.read_csv(file_path)

# -------------------------------
# 2. 전처리 및 기준 분류
# -------------------------------
df = df[['file', 'mean_sim', 'difficulty_median', 'Kread', '표지어_총합']].copy()

# KReaD 점수를 binary 난이도로 변환 (1799 이하: Easy / 1800 이상: Hard)
df['Kread_binary'] = df['Kread'].apply(lambda x: 'Easy' if x <= 1799 else 'Hard')

# 유사도 기준은 difficulty_median 사용 (Easy, Hard만 비교 대상)
def is_disagree(row):
    kread = row['Kread_binary']
    sim = row['difficulty_median']
    return (kread != sim) and (kread in ['Easy', 'Hard']) and (sim in ['Easy', 'Hard'])

disagree_df = df[df.apply(is_disagree, axis=1)]

# -------------------------------
# 3. 표지어 평균 계산
# -------------------------------
# 전체 기준 평균
overall_mean = df['표지어_총합'].mean()

# KReaD 기준 난이도별 평균
kread_easy_mean = df[df['Kread_binary'] == 'Easy']['표지어_총합'].mean()
kread_hard_mean = df[df['Kread_binary'] == 'Hard']['표지어_총합'].mean()

# 유사도 기준 난이도별 평균
sim_easy_mean = df[df['difficulty_median'] == 'Easy']['표지어_총합'].mean()
sim_hard_mean = df[df['difficulty_median'] == 'Hard']['표지어_총합'].mean()

# 상반된 지문 10개에서의 표지어 평균
disagree_marker_mean = disagree_df['표지어_총합'].mean()

# -------------------------------
# 4. 어떤 기준에 더 가까운가 판단
# -------------------------------
# 전체 지문 수 출력
print(f"📁 상반 지문 수: {len(disagree_df)}")

# 표지어 평균 비교
print("\n📊 표지어 평균 비교")
print(f" - 전체 평균: {overall_mean:.2f}")
print(f" - KReaD Easy 평균: {kread_easy_mean:.2f}")
print(f" - KReaD Hard 평균: {kread_hard_mean:.2f}")
print(f" - 유사도 Easy 평균: {sim_easy_mean:.2f}")
print(f" - 유사도 Hard 평균: {sim_hard_mean:.2f}")
print(f" - 상반 지문 평균: {disagree_marker_mean:.2f}")

# 거리 계산 (절댓값 차이)
dist_to_kread = min(abs(disagree_marker_mean - kread_easy_mean), abs(disagree_marker_mean - kread_hard_mean))
dist_to_sim = min(abs(disagree_marker_mean - sim_easy_mean), abs(disagree_marker_mean - sim_hard_mean))

closer_to = "KReaD 기준" if dist_to_kread < dist_to_sim else "유사도 기준"
print(f"\n✅ 상반 지문들의 표지어 개수 평균은 ▶ {closer_to}에 더 가까움")
