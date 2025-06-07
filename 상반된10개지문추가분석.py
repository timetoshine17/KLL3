import pandas as pd
import numpy as np

# -------------------------------
# 1. 파일 불러오기 (CSV 파일 직접 읽기)
# -------------------------------
# 파일 경로를 CSV로 수정하고, 실제 업로드된 파일 사용
df = pd.read_csv(r"C:\Users\82109\Desktop\고려대\2025-1\한국어정보처리\상반 지문 추가분석.csv", encoding='cp949')


# -------------------------------
# 2. 전체 표지어 평균 계산
# -------------------------------
overall_mean = df['표지어_총합'].mean()

# -------------------------------
# 3. 난이도별 표지어 평균 계산
# -------------------------------
# KReaD 기준
kread_easy = df[df['Kread_binary'] == 'Easy']
kread_hard = df[df['Kread_binary'] == 'Hard']

print(f"KReaD 기준 분포:")
print(f" - Easy 샘플 수: {len(kread_easy)}")
print(f" - Hard 샘플 수: {len(kread_hard)}")

if len(kread_easy) > 0:
    kread_easy_mean = kread_easy['표지어_총합'].mean()
    print(f" - Easy 표지어 평균: {round(kread_easy_mean, 2)}")
else:
    print(" - Easy 샘플이 없습니다.")

if len(kread_hard) > 0:
    kread_hard_mean = kread_hard['표지어_총합'].mean()
    print(f" - Hard 표지어 평균: {round(kread_hard_mean, 2)}")
else:
    print(" - Hard 샘플이 없습니다.")

# 유사도 기준
sim_easy = df[df['difficulty_median'] == 'Easy']
sim_hard = df[df['difficulty_median'] == 'Hard']

print(f"유사도(difficulty_median) 기준 분포:")
print(f" - Easy 샘플 수: {len(sim_easy)}")
print(f" - Hard 샘플 수: {len(sim_hard)}")

if len(sim_easy) > 0:
    sim_easy_mean = sim_easy['표지어_총합'].mean()
    print(f" - Easy 표지어 평균: {round(sim_easy_mean, 2)}")
else:
    print(" - Easy 샘플이 없습니다.")

if len(sim_hard) > 0:
    sim_hard_mean = sim_hard['표지어_총합'].mean()
    print(f" - Hard 표지어 평균: {round(sim_hard_mean, 2)}")
else:
    print(" - Hard 샘플이 없습니다.")

# -------------------------------
# 4. 상세 분석 결과 출력
# -------------------------------
print(f"=== 분석 결과 요약 ===")
print(f"전체 표지어 평균: {round(overall_mean, 2)}")
print(f"전체 샘플 수: {len(df)}")


'''
KReaD 기준 분포:
 - Easy 샘플 수: 7
 - Hard 샘플 수: 5
 - Easy 표지어 평균: 9.57
 - Hard 표지어 평균: 7.6
유사도(difficulty_median) 기준 분포:
 - Easy 샘플 수: 5
 - Hard 샘플 수: 7
 - Easy 표지어 평균: 7.6
 - Hard 표지어 평균: 9.57
=== 분석 결과 요약 ===
전체 표지어 평균: 8.75
전체 샘플 수: 12
''' 
