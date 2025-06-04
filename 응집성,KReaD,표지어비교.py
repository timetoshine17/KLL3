import pandas as pd

# -------------------------------
# 1. 파일 불러오기 
# -------------------------------
file_path = r"C:\Users\82109\Desktop\고려대\2025-1\한국어정보처리\Difficulty_Classification__Median___Quartile_.cell"  # 또는 .csv로 수정
df = pd.read_excel(file_path)

# -------------------------------
# 2. 필요한 열만 추출
# -------------------------------
df = df[['file', 'mean_sim', 'difficulty_median', 'Kread', '표지어_총합']].copy()

# -------------------------------
# 3. KReaD 점수 이진 분류 (1799 이하 = Easy, 1800 이상 = Hard)
# -------------------------------
def classify_kread_binary(score):
    return 'Easy' if score <= 1799 else 'Hard'

df['Kread_binary'] = df['Kread'].apply(classify_kread_binary)

# -------------------------------
# 4. 유사도 기준은 difficulty_median 사용 (Medium 제거 또는 비교에서 제외 가능)
# 이번에는 Medium도 포함하고 Easy/Hard가 상반될 때만 추출
# -------------------------------
def is_disagree(row):
    # 서로 다른 범주이면서, 둘 다 Easy 또는 Hard인 경우만 상반 판단
    kread = row['Kread_binary']
    sim = row['difficulty_median']
    return (kread != sim) and (kread in ['Easy', 'Hard']) and (sim in ['Easy', 'Hard'])

disagree_df = df[df.apply(is_disagree, axis=1)]

# -------------------------------
# 5. 상반된 지문을 CSV로 저장
# -------------------------------
disagree_df.to_csv("유사도-KReaD_상반_지문_binary.csv", index=False)

# -------------------------------
# 6. 전체 표지어 평균 계산
# -------------------------------
overall_mean = df['표지어_총합'].mean()

# -------------------------------
# 7. 난이도별 표지어 평균 계산
# -------------------------------
# KReaD 기준
kread_easy = df[df['Kread_binary'] == 'Easy']
kread_hard = df[df['Kread_binary'] == 'Hard']
kread_easy_mean = kread_easy['표지어_총합'].mean()
kread_hard_mean = kread_hard['표지어_총합'].mean()

# 유사도 기준
sim_easy = df[df['difficulty_median'] == 'Easy']
sim_hard = df[df['difficulty_median'] == 'Hard']
sim_easy_mean = sim_easy['표지어_총합'].mean()
sim_hard_mean = sim_hard['표지어_총합'].mean()

# -------------------------------
# 8. 결과 출력
# -------------------------------
print("📊 전체 표지어 평균:", round(overall_mean, 2))
print("\n📘 KReaD 기준 표지어 평균")
print(" - Easy:", round(kread_easy_mean, 2))
print(" - Hard:", round(kread_hard_mean, 2))

print("\n📗 유사도 기준 표지어 평균")
print(" - Easy:", round(sim_easy_mean, 2))
print(" - Hard:", round(sim_hard_mean, 2))

print("\n📁 상반된 지문 수:", len(disagree_df))
print("🔗 저장된 파일: 유사도-KReaD_상반_지문_binary.csv")
