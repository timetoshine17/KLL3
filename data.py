# -*- coding: utf-8 -*-
import os
from sentence_transformers import SentenceTransformer, util

# 1. 모델 로드
model = SentenceTransformer("nlpai-lab/KURE-v1")

# 2. 대상 폴더 설정
target_folder = r"C:\Users\User\OneDrive\바탕 화면\2025\corpus"



# 3. 폴더 내 모든 .txt 파일 순회
for filename in os.listdir(target_folder):
    if filename.endswith(".txt"):
        full_path = os.path.join(target_folder, filename)
        print(f"\n📄 파일 처리 중: {filename}")

        # 4. 텍스트 파일 열기
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
        except UnicodeDecodeError:
            print("⚠️ 파일 인코딩 오류: UTF-8 아님 (생략)")
            continue

        # 5. 문장 분리 (여기선 줄바꿈 기준, 필요시 정교하게 개선 가능)
        sentences = [line.strip() for line in content.split('\n') if line.strip()]
        if len(sentences) < 2:
            print("⚠️ 문장이 부족해 유사도 분석 생략")
            continue

        # 6. 임베딩 생성
        embeddings = model.encode(sentences, convert_to_tensor=True)

        # 7. 유사도 계산
        similarities = util.cos_sim(embeddings, embeddings)

        # 8. 결과 출력
        print(f"✅ 문장 수: {len(sentences)}")
        print(f"✅ 임베딩 shape: {embeddings.shape}")
        print("🧠 유사도 행렬:")
        print(similarities)
