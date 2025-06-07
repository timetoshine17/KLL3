import pandas as pd
import numpy as np

# -------------------------------
# 1. Excel 파일 불러오기
# -------------------------------
# 파일 경로를 실제 Excel 파일 경로로 수정하세요
excel_file_path = r"C:\Users\82109\Desktop\고려대\2025-1\한국어정보처리\유사도-KReaD-표지어-등급컷 통합.xlsx"  # 실제 파일 경로로 변경 필요

df = pd.read_excel(excel_file_path)
print("Excel 파일을 성공적으로 불러왔습니다.")



# -------------------------------
# 2. KReaD 점수 이진 분류 (1799 이하 = Easy, 1800 이상 = Hard)
# -------------------------------
def classify_kread_binary(score):
    """KReaD 점수를 Easy/Hard로 분류"""
    if pd.isna(score):
        return 'Unknown'
    return 'Easy' if score <= 1799 else 'Hard'

df['Kread_binary'] = df['Kread'].apply(classify_kread_binary)

print(f"KReaD 분류 결과:")
print(df['Kread_binary'].value_counts())

# -------------------------------
# 3. CSV 파일로 저장
# -------------------------------
csv_output_path = "유사도-KReaD-표지어-등급컷 통합.csv"
df.to_csv(csv_output_path, index=False, encoding='utf-8-sig')
print(f"CSV 파일 저장: {csv_output_path}")

# -------------------------------
# 4. 전체 표지어 평균 계산
# -------------------------------
# 결측치 제거
df_clean = df.dropna(subset=['표지어_총합'])
overall_mean = df_clean['표지어_총합'].mean()

print(f"=== 전체 통계 ===")
print(f"전체 샘플 수: {len(df_clean)}")
print(f"전체 표지어 평균: {round(overall_mean, 2)}")

# -------------------------------
# 5. KReaD 기준 난이도별 표지어 평균 계산
# -------------------------------
kread_easy = df_clean[df_clean['Kread_binary'] == 'Easy']
kread_hard = df_clean[df_clean['Kread_binary'] == 'Hard']

print(f"KReaD Easy 샘플 수: {len(kread_easy)}")
print(f"KReaD Hard 샘플 수: {len(kread_hard)}")

if len(kread_easy) > 0:
    kread_easy_mean = kread_easy['표지어_총합'].mean()
    kread_easy_std = kread_easy['표지어_총합'].std()
    print(f"KReaD Easy 표지어 평균: {round(kread_easy_mean, 2)} (±{round(kread_easy_std, 2)})")

if len(kread_hard) > 0:
    kread_hard_mean = kread_hard['표지어_총합'].mean()
    kread_hard_std = kread_hard['표지어_총합'].std()
    print(f"KReaD Hard 표지어 평균: {round(kread_hard_mean, 2)} (±{round(kread_hard_std, 2)})")


# -------------------------------
# 6. 유사도(difficulty_median) 기준 난이도별 표지어 평균 계산
# -------------------------------
# difficulty_median의 고유값 확인
print(f"difficulty_median 값들: {df_clean['difficulty_median'].unique()}")

sim_easy = df_clean[df_clean['difficulty_median'] == 'Easy']
sim_hard = df_clean[df_clean['difficulty_median'] == 'Hard']

print(f"유사도 Easy 샘플 수: {len(sim_easy)}")
print(f"유사도 Hard 샘플 수: {len(sim_hard)}")

if len(sim_easy) > 0:
    sim_easy_mean = sim_easy['표지어_총합'].mean()
    sim_easy_std = sim_easy['표지어_총합'].std()
    print(f"유사도 Easy 표지어 평균: {round(sim_easy_mean, 2)} (±{round(sim_easy_std, 2)})")


if len(sim_hard) > 0:
    sim_hard_mean = sim_hard['표지어_총합'].mean()
    sim_hard_std = sim_hard['표지어_총합'].std()
    print(f"유사도 Hard 표지어 평균: {round(sim_hard_mean, 2)} (±{round(sim_hard_std, 2)})")


# -------------------------------
# 7. 두 분류 방법 비교 분석
# -------------------------------
# 일치/불일치 분석
agreement_df = df_clean[df_clean['Kread_binary'] == df_clean['difficulty_median']]
disagreement_df = df_clean[df_clean['Kread_binary'] != df_clean['difficulty_median']]

print(f"일치하는 샘플 수: {len(agreement_df)}")
print(f"불일치하는 샘플 수: {len(disagreement_df)}")

if len(disagreement_df) > 0:
    print(f"불일치 샘플들:")
    for idx, row in disagreement_df.iterrows():
        print(f"  - {row['file']}: KReaD={row['Kread_binary']}, 유사도={row['difficulty_median']}, 표지어={row['표지어_총합']}")

# -------------------------------
# 8. 상세 통계 결과 저장
# -------------------------------
results_summary = {
    "전체_샘플수": len(df_clean),
    "전체_표지어_평균": round(overall_mean, 2),
    "KReaD_Easy_샘플수": len(kread_easy),
    "KReaD_Easy_표지어_평균": round(kread_easy_mean, 2) if len(kread_easy) > 0 else None,
    "KReaD_Hard_샘플수": len(kread_hard),
    "KReaD_Hard_표지어_평균": round(kread_hard_mean, 2) if len(kread_hard) > 0 else None,
    "유사도_Easy_샘플수": len(sim_easy),
    "유사도_Easy_표지어_평균": round(sim_easy_mean, 2) if len(sim_easy) > 0 else None,
    "유사도_Hard_샘플수": len(sim_hard),
    "유사도_Hard_표지어_평균": round(sim_hard_mean, 2) if len(sim_hard) > 0 else None,
    "분류_일치_샘플수": len(agreement_df),
    "분류_불일치_샘플수": len(disagreement_df)
}



'''
결과

Excel 파일을 성공적으로 불러왔습니다.
KReaD 분류 결과:
Kread_binary
Easy    16
Hard    11
Name: count, dtype: int64
CSV 파일 저장: 유사도-KReaD-표지어-등급컷 통합.csv
=== 전체 통계 ===
전체 샘플 수: 27
전체 표지어 평균: 8.89
KReaD Easy 샘플 수: 16
KReaD Hard 샘플 수: 11
KReaD Easy 표지어 평균: 9.38 (±6.0)
KReaD Hard 표지어 평균: 8.18 (±3.76)
difficulty_median 값들: ['Hard' 'Easy']
유사도 Easy 샘플 수: 14
유사도 Hard 샘플 수: 13
유사도 Easy 표지어 평균: 8.64 (±4.7)
유사도 Hard 표지어 평균: 9.15 (±5.79)
difficulty_median 값들: ['Hard' 'Easy']
유사도 Easy 샘플 수: 14
유사도 Hard 샘플 수: 13
유사도 Easy 표지어 평균: 8.64 (±4.7)
유사도 Hard 표지어 평균: 9.15 (±5.79)
유사도 Hard 표지어 평균: 9.15 (±5.79)
일치하는 샘플 수: 15
불일치하는 샘플 수: 12
불일치 샘플들:
  - 20210603: KReaD=Easy, 유사도=Hard, 표지어=5
불일치하는 샘플 수: 12
불일치 샘플들:
  - 20210603: KReaD=Easy, 유사도=Hard, 표지어=5
불일치 샘플들:
  - 20210603: KReaD=Easy, 유사도=Hard, 표지어=5
  - 20210901: KReaD=Hard, 유사도=Easy, 표지어=8
  - 20210902: KReaD=Hard, 유사도=Easy, 표지어=13
  - 20210903: KReaD=Hard, 유사도=Easy, 표지어=6
  - 20211102: KReaD=Easy, 유사도=Hard, 표지어=4
  - 20211103: KReaD=Hard, 유사도=Easy, 표지어=4
  - 20220601: KReaD=Easy, 유사도=Hard, 표지어=25
  - 20220602: KReaD=Easy, 유사도=Hard, 표지어=8
  - 20230602: KReaD=Easy, 유사도=Hard, 표지어=10
  - 20230603: KReaD=Easy, 유사도=Hard, 표지어=6
  - 20231101: KReaD=Hard, 유사도=Easy, 표지어=7
  - 20230602: KReaD=Easy, 유사도=Hard, 표지어=10
  - 20230603: KReaD=Easy, 유사도=Hard, 표지어=6
  - 20231101: KReaD=Hard, 유사도=Easy, 표지어=7
  - 20231103: KReaD=Easy, 유사도=Hard, 표지어=9
  '''
