import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# CSV 파일 로드
try:
    df_grade = pd.read_csv(r'C:\Users\82109\Desktop\고려대\2025-1\한국어정보처리\팀플\등급컷.csv')  # 등급 데이터 파일
    df_similarity = pd.read_csv(r'C:\Users\82109\Desktop\고려대\2025-1\한국어정보처리\팀플\유사도-KReaD-표지어-등급컷 통합.csv')  # 유사도 데이터 파일
    print("CSV 파일 로드 완료")
except FileNotFoundError as e:
    print(f"파일을 찾을 수 없습니다: {e}")
    print("다음 파일들이 필요합니다:")
    print("1. grade_data.csv - 등급 데이터")
    print("2. similarity_data.csv - 유사도 데이터")
    exit()

# 데이터 확인
print("\n=== 데이터 정보 ===")
print(f"등급 데이터 shape: {df_grade.shape}")
print(f"유사도 데이터 shape: {df_similarity.shape}")
print("\n등급 데이터 컬럼:", df_grade.columns.tolist())
print("유사도 데이터 컬럼:", df_similarity.columns.tolist())

# 날짜 형식 통일 및 데이터 병합
df_grade['date_key'] = df_grade['날짜'].astype(str)
df_similarity['date_key'] = df_similarity['file'].astype(str).str[:6]

# 등급별 데이터 분리
df_grade_1 = df_grade[df_grade['등급'] == '1등급'].copy()
df_grade_2 = df_grade[df_grade['등급'] == '2등급'].copy()
df_grade_3 = df_grade[df_grade['등급'] == '3등급'].copy()

# 병합 (각 등급별로)
merged_1 = pd.merge(df_similarity, df_grade_1, on='date_key', how='inner')
merged_2 = pd.merge(df_similarity, df_grade_2, on='date_key', how='inner')
merged_3 = pd.merge(df_similarity, df_grade_3, on='date_key', how='inner')

print(f"\n병합 결과:")
print(f"1등급 데이터: {len(merged_1)}개")
print(f"2등급 데이터: {len(merged_2)}개")
print(f"3등급 데이터: {len(merged_3)}개")

# 날짜 변환 함수
def convert_date(date_str):
    return datetime.strptime(str(date_str), '%Y%m%d')

# 각 데이터프레임에 날짜 컬럼 추가
for df in [merged_1, merged_2, merged_3]:
    df['date'] = df['file'].apply(convert_date)
    df['원점수'] = df['원점수(화작/언매)'].astype(float)

# 분류별 평균 등급컷 출력
print("=" * 60)
print("분류별 평균 등급컷 분석")
print("=" * 60)

for grade_num, df in enumerate([merged_1, merged_2, merged_3], 1):
    print(f"\n【{grade_num}등급】")
    
    # 유사도 기반 분류
    sim_easy = df[df['difficulty_median'] == 'Easy']['원점수']
    sim_hard = df[df['difficulty_median'] == 'Hard']['원점수']
    
    print(f"유사도 기반:")
    print(f"  Easy 평균: {sim_easy.mean():.1f}점 ({len(sim_easy)}개)")
    print(f"  Hard 평균: {sim_hard.mean():.1f}점 ({len(sim_hard)}개)")
    
    # KReaD 기반 분류
    kread_easy = df[df['Kread_binary'] == 'Easy']['원점수']
    kread_hard = df[df['Kread_binary'] == 'Hard']['원점수']
    
    print(f"KReaD 기반:")
    print(f"  Easy 평균: {kread_easy.mean():.1f}점 ({len(kread_easy)}개)")
    print(f"  Hard 평균: {kread_hard.mean():.1f}점 ({len(kread_hard)}개)")
    
    # 차이 분석
    sim_diff = sim_easy.mean() - sim_hard.mean()
    kread_diff = kread_easy.mean() - kread_hard.mean()
    
    print(f"Easy-Hard 차이:")
    print(f"  유사도: {sim_diff:+.1f}점")
    print(f"  KReaD: {kread_diff:+.1f}점")

print("\n" + "=" * 60)

# 시각화 - 포인트 분리 개선 버전
fig, axes = plt.subplots(3, 1, figsize=(18, 14))
fig.suptitle('시간별 등급컷 추이 및 난이도 분류 결과', fontsize=16, fontweight='bold')

grade_names = ['1등급', '2등급', '3등급']
colors = ['#2E86AB', '#A23B72', '#F18F01']

for i, (df, grade_name, color) in enumerate(zip([merged_1, merged_2, merged_3], grade_names, colors)):
    ax = axes[i]
    
    # 데이터 정렬
    df_sorted = df.sort_values('date')
    
    # 고유한 날짜들을 추출하고 숫자 인덱스로 변환
    unique_dates = df_sorted['date'].unique()
    date_to_x = {date: i for i, date in enumerate(unique_dates)}
    
    # 기본 선 그래프 (등급컷 추이) - 숫자 좌표 사용
    x_positions = [date_to_x[date] for date in df_sorted['date']]
    ax.plot(x_positions, df_sorted['원점수'], 
            color=color, linewidth=3, marker='o', markersize=10, 
            label=f'{grade_name} 등급컷', zorder=3)
    
    # 분류별로 분리하여 표시
    for idx, row in df_sorted.iterrows():
        base_x = date_to_x[row['date']]
        score = row['원점수']
        
        # 유사도 분류 (왼쪽으로 오프셋)
        sim_x = base_x - 0.15  # 더 작은 오프셋으로 가깝게 배치
        sim_color = '#FF3333' if row['difficulty_median'] == 'Hard' else '#33CCCC'
        sim_marker = '^' if row['difficulty_median'] == 'Hard' else 'v'
        ax.scatter(sim_x, score, 
                  c=sim_color, s=150, marker=sim_marker, 
                  alpha=0.9, edgecolors='white', linewidth=2, zorder=4)
        
        # KReaD 분류 (오른쪽으로 오프셋)
        kread_x = base_x + 0.15  # 더 작은 오프셋으로 가깝게 배치
        kread_color = '#CC3333' if row['Kread_binary'] == 'Hard' else '#33CC33'
        kread_marker = 's' if row['Kread_binary'] == 'Hard' else 'D'
        ax.scatter(kread_x, score, 
                  c=kread_color, s=150, marker=kread_marker, 
                  alpha=0.9, edgecolors='white', linewidth=2, zorder=4)
    
    # 축 설정
    ax.set_title(f'{grade_name} 등급컷 추이', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('등급컷 (점)', fontsize=12)
    ax.grid(True, alpha=0.3, zorder=0)
    
    # y축 범위 조정
    y_min = df_sorted['원점수'].min() - 3
    y_max = df_sorted['원점수'].max() + 3
    ax.set_ylim(y_min, y_max)
    
    # x축을 날짜 레이블로 설정
    ax.set_xticks(range(len(unique_dates)))
    ax.set_xticklabels([date.strftime('%Y-%m') for date in unique_dates], rotation=45)
    ax.set_xlim(-0.5, len(unique_dates) - 0.5)  # x축 범위 설정
    
    # 범례 (마지막 서브플롯에만)
    if i == 2:
        legend_elements = [
            plt.Line2D([0], [0], color=color, linewidth=3, marker='o', 
                      markersize=10, label='등급컷 추이'),
            plt.Line2D([0], [0], color='#FF3333', linewidth=0, marker='^', 
                      markersize=12, label='유사도 Hard', markeredgecolor='white', markeredgewidth=2),
            plt.Line2D([0], [0], color='#33CCCC', linewidth=0, marker='v', 
                      markersize=12, label='유사도 Easy', markeredgecolor='white', markeredgewidth=2),
            plt.Line2D([0], [0], color='#CC3333', linewidth=0, marker='s', 
                      markersize=12, label='KReaD Hard', markeredgecolor='white', markeredgewidth=2),
            plt.Line2D([0], [0], color='#33CC33', linewidth=0, marker='D', 
                      markersize=12, label='KReaD Easy', markeredgecolor='white', markeredgewidth=2)
        ]
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), 
                 fontsize=12, frameon=True, fancybox=True, shadow=True)

plt.xlabel('시기', fontsize=12)
plt.tight_layout()
plt.show()

# 정확도 분석 (1등급 기준)
print("\n" + "=" * 60)
print("분류 정확도 분석 (1등급 기준)")
print("=" * 60)

# 85점을 기준으로 어려움/쉬움 판단
threshold = 85
df_1 = merged_1.copy()
df_1['actual_hard'] = df_1['원점수'] < threshold
df_1['sim_correct'] = df_1['actual_hard'] == (df_1['difficulty_median'] == 'Hard')
df_1['kread_correct'] = df_1['actual_hard'] == (df_1['Kread_binary'] == 'Hard')

sim_accuracy = df_1['sim_correct'].mean() * 100
kread_accuracy = df_1['kread_correct'].mean() * 100

print(f"기준점: {threshold}점 미만을 '어려움'으로 분류")
print(f"유사도 기반 정확도: {sim_accuracy:.1f}%")
print(f"KReaD 기반 정확도: {kread_accuracy:.1f}%")

'''
결과
CSV 파일 로드 완료

=== 데이터 정보 ===
등급 데이터 shape: (27, 5)
유사도 데이터 shape: (27, 5)

등급 데이터 컬럼: ['날짜', '등급', '원점수(화작/언매)', '표준점수', '백분위']
유사도 데이터 컬럼: ['file', 'difficulty_median', 'Kread', '표지어_총합', 'Kread_binary']

병합 결과:
1등급 데이터: 27개
2등급 데이터: 27개
3등급 데이터: 27개
============================================================
분류별 평균 등급컷 분석
============================================================

【1등급】
유사도 기반:
  Easy 평균: 90.5점 (14개)
  Hard 평균: 89.1점 (13개)
KReaD 기반:
  Easy 평균: 89.6점 (16개)
  Hard 평균: 90.1점 (11개)
Easy-Hard 차이:
  유사도: +1.4점
  KReaD: -0.5점

【2등급】
유사도 기반:
  Easy 평균: 84.7점 (14개)
  Hard 평균: 82.5점 (13개)
KReaD 기반:
  Easy 평균: 83.3점 (16개)
  Hard 평균: 84.0점 (11개)
Easy-Hard 차이:
  유사도: +2.2점
  KReaD: -0.7점

【3등급】
유사도 기반:
  Easy 평균: 77.5점 (14개)
  Hard 평균: 74.5점 (13개)
KReaD 기반:
  Easy 평균: 75.8점 (16개)
  Hard 평균: 76.5점 (11개)
Easy-Hard 차이:
  유사도: +3.1점
  KReaD: -0.7점

============================================================

============================================================
분류 정확도 분석 (1등급 기준)
============================================================
기준점: 85점 미만을 '어려움'으로 분류
유사도 기반 정확도: 51.9%
KReaD 기반 정확도: 59.3%
'''
