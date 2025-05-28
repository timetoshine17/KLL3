import torch
import re

# 원본 문자열
tensor_str = """
[[1.0000, 0.6510, 0.6814]
 [0.6510, 1.0000, 0.6705]
 [0.6814, 0.6705, 1.0000]]
"""

# 1) 각 행에서 숫자를 추출해 2D 리스트 생성
rows = []
for line in tensor_str.strip().splitlines():
    nums = [float(x) for x in re.findall(r'\d+\.\d+', line)]
    rows.append(nums)

# 2) torch.Tensor로 변환
mat = torch.tensor(rows)

# 3) mean_offdiag 함수 정의
def mean_offdiag(mat: torch.Tensor) -> float:
    """대각선을 뺀 평균 유사도 계산"""
    n = mat.size(0)
    return ((mat.sum() - mat.trace()) / (n * (n - 1))).item()

# 4) 계산 및 출력
result = mean_offdiag(mat)
print(f"Mean off-diagonal similarity: {result:.6f}")
