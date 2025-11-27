# High-Resolution Data Processing for Zero-Shot Super-Resolution

이 문서는 고해상도 데이터를 저해상도에서 학습된 normalizer로 전처리하는 방법을 설명합니다.

## 핵심 원칙

**Zero-shot super-resolution에서는 저해상도 훈련 시 사용한 normalizer와 동일한 normalizer를 고해상도 데이터에 적용해야 합니다.**

- ✅ **올바름**: 저해상도 normalizer를 고해상도 데이터에 적용
- ❌ **틀림**: 고해상도 데이터로 새로운 normalizer 학습

## 파일 구조

```
src/preprocessing/
├── preprocessing_merge.py           # 저해상도 데이터 병합 및 정규화
├── preprocessing_merge_hr.py        # 고해상도 데이터 병합 및 정규화 (NEW!)
├── preprocessing_normalizer.py      # Normalizer 구현
├── preprocessing_normalizer_hr.py   # 고해상도 적용 함수 (NEW!)
└── README_HR.md                     # 이 문서
```

## 워크플로우

### 1단계: 저해상도 데이터로 훈련 (기존)

```bash
cd /home/geofluids/research/FNO

# 저해상도 전처리
python src/preprocessing/preprocessing_merge.py
# 입력 시 output mode 선택: log (권장)

# 생성 파일:
#   - merged_U_log_normalized.pt           # 정규화된 훈련 데이터
#   - channel_normalizer_log.pkl           # ← 이것을 반드시 보관!
#   - normalization_check/                 # 통계 및 분포 확인
```

### 2단계: 저해상도로 모델 훈련

```bash
# FNO 훈련
cd src/FNO
python FNO.py

# CONFIG 설정:
# - MERGED_PT_PATH: './src/preprocessing/merged_U_log_normalized.pt'
# - CHANNEL_NORMALIZER_PATH: './src/preprocessing/channel_normalizer_log.pkl'
```

### 3단계: 고해상도 데이터 준비

고해상도 PFLOTRAN 시뮬레이션 데이터를 전처리합니다:

```bash
cd /home/geofluids/research/FNO

# 고해상도 데이터 전처리 (preprocessing_revised.py)
# 출력 파일명에 '_hr' 포함 권장
# 예: input_output_com1_hr.pt, input_output_com2_hr.pt, ...
```

### 4단계: 고해상도 데이터 병합 및 정규화

```bash
cd /home/geofluids/research/FNO

# 고해상도 데이터 병합 및 정규화
python src/preprocessing/preprocessing_merge_hr.py

# 실행 흐름:
# 1. 고해상도 파일 자동 검색 (input_output_*_hr.pt 패턴)
# 2. 파일 병합 → merged_U_raw_hr.pt
# 3. 저해상도 normalizer 경로 입력 요청
#    기본값: ./src/preprocessing/channel_normalizer_log.pkl
# 4. 동일한 normalizer 적용 → merged_U_log_normalized_hr.pt
```

**실행 예시:**

```
======================================================================
High-Resolution Data Merge and Normalization
======================================================================

Step 1: Merging high-resolution data shards...
----------------------------------------------------------------------
Searching for high-resolution files...
Found 3 file(s):
  - ./src/preprocessing/input_output_com1_hr.pt
  - ./src/preprocessing/input_output_com2_hr.pt
  - ./src/preprocessing/input_output_com3_hr.pt

[OK] Merge complete: 3 shard(s) → ./src/preprocessing/merged_U_raw_hr.pt
Final shapes: x(150, 11, 64, 64, 20) y(150, 1, 64, 64, 20)
Spatial resolution: 64 × 64  ← 저해상도(32×32)의 2배!

======================================================================
Step 2: Applying Pre-trained Normalizer to High-Resolution Data
======================================================================

⚠️  IMPORTANT: Use the normalizer from LOW-RESOLUTION training data!

Enter normalizer path [./src/preprocessing/channel_normalizer_log.pkl]:
Inferred output mode from normalizer: 'log'

Step 1: Loading pre-trained normalizer...
   Output mode: log

Step 2: Loading high-resolution raw data...
   HR input shape:  (150, 11, 64, 64, 20)
   HR output shape: (150, 1, 64, 64, 20)

Step 3: Applying pre-trained normalizer to HR data...
   (Using SAME statistics as low-resolution training data)

Step 4: Saving normalized HR data...

======================================================================
SUCCESS: High-resolution data normalized!
======================================================================

⚠️  Remember: Use the SAME normalizer for inverse transform!
======================================================================
```

### 5단계: Zero-Shot Super-Resolution 예측

```python
# FNO 모델로 고해상도 예측
import torch
import pickle
from neuraloperator.neuralop.models import TFNO

# 1. 저해상도로 훈련된 모델 로드
model = TFNO(
    n_modes=(19, 6, 4),  # 저해상도 훈련 시 사용한 설정
    hidden_channels=48,
    n_layers=6,
    # ... 기타 파라미터
)
model.load_state_dict(torch.load('./src/FNO/output_pure/final/best_model_state_dict.pt'))
model.eval()
model = model.to('cuda')

# 2. 저해상도 normalizer 로드 (중요!)
with open('./src/preprocessing/channel_normalizer_log.pkl', 'rb') as f:
    normalizer = pickle.load(f)
normalizer = normalizer.to('cuda')

# 3. 고해상도 정규화된 데이터 로드
hr_data = torch.load('./src/preprocessing/merged_U_log_normalized_hr.pt')
x_hr = hr_data['x'].to('cuda')  # (150, 11, 64, 64, 19)

# 4. Zero-shot super-resolution 예측
#    해상도 차이는 FNO가 자동으로 처리!
with torch.no_grad():
    pred_hr_normalized = model(x_hr)  # (150, 1, 64, 64, 19)

# 5. 동일한 normalizer로 역변환
pred_hr_physical = normalizer.inverse_transform_output_to_raw(pred_hr_normalized)

print(f"Input resolution:  {x_hr.shape[2]} × {x_hr.shape[3]}")     # 64 × 64
print(f"Output resolution: {pred_hr_physical.shape[2]} × {pred_hr_physical.shape[3]}")  # 64 × 64
print(f"Prediction range: [{pred_hr_physical.min():.2e}, {pred_hr_physical.max():.2e}]")
```

## 파일명 규칙

### 저해상도 (Low-Resolution)
- Raw 데이터: `merged_U_raw.pt`
- 정규화 데이터: `merged_U_log_normalized.pt` (mode에 따라 변경)
- Normalizer: `channel_normalizer_log.pkl`

### 고해상도 (High-Resolution)
- Raw 데이터: `merged_U_raw_hr.pt`
- 정규화 데이터: `merged_U_log_normalized_hr.pt`
- Normalizer: **저해상도와 동일한 파일 사용!** (`channel_normalizer_log.pkl`)

## 주의사항

### ✅ 올바른 사용법

1. **저해상도로 훈련**
   - `preprocessing_merge.py` 실행
   - `channel_normalizer_log.pkl` 생성 및 보관

2. **고해상도 적용**
   - `preprocessing_merge_hr.py` 실행
   - **저해상도 normalizer 경로 입력**

3. **예측 후 역변환**
   - **저해상도 normalizer로 inverse transform**

### ❌ 흔한 실수

1. ❌ 고해상도 데이터로 새로운 normalizer 생성
   ```python
   # 틀린 방법!
   normalizer_hr = ChannelWiseNormalizer()
   normalizer_hr.fit(x_hr, y_hr)  # 통계량이 달라짐!
   ```

2. ❌ 다른 output_mode의 normalizer 사용
   ```python
   # 틀린 방법!
   # 훈련: channel_normalizer_log.pkl
   # 예측: channel_normalizer_delta.pkl  # mode가 다름!
   ```

3. ❌ Normalizer 없이 raw 데이터 직접 입력
   ```python
   # 틀린 방법!
   x_hr_raw = torch.load('merged_U_raw_hr.pt')['x']
   pred = model(x_hr_raw)  # 정규화되지 않은 데이터!
   ```

## FNO가 Zero-Shot Super-Resolution이 가능한 이유

1. **Fourier 변환 기반**: 주파수 도메인에서 작동하므로 공간 해상도에 독립적
2. **Mode 수 제한**: `n_modes`로 지정된 주파수만 학습하므로, 입력 해상도 > `n_modes * 2` 범위에서 일반화
3. **Continuous Operator**: 연속 함수 공간 간의 매핑을 학습하므로 이산화 격자에 의존하지 않음

### n_modes 설정 가이드

```python
# 저해상도 훈련 데이터: 32 × 32 × 20
n_modes = (19, 6, 4)  # 각 차원의 절반 정도

# Zero-shot 가능 범위 (대략적):
# - 최소: n_modes * 2 ≈ (38, 12, 8)
# - 최대: n_modes * 4 ≈ (76, 24, 16)

# 2배 해상도 (64 × 64 × 20): ✅ 가능
# 4배 해상도 (128 × 128 × 20): ⚠️ 성능 저하 가능성
```

## 문제 해결

### Q1: "Normalizer not found" 오류

**원인**: 저해상도 normalizer 파일이 없음

**해결**:
```bash
# 저해상도 데이터로 먼저 전처리 필요
python src/preprocessing/preprocessing_merge.py
```

### Q2: "Channel count mismatch" 오류

**원인**: 저해상도와 고해상도 데이터의 채널 수가 다름

**해결**:
- 두 데이터 모두 동일한 `preprocessing_revised.py`로 전처리했는지 확인
- 저해상도: (N, 11, 32, 32, 20)
- 고해상도: (N, 11, 64, 64, 20) ← 공간 차원만 다름

### Q3: 예측 결과가 이상함

**가능한 원인**:
1. 다른 normalizer 사용
2. n_modes가 너무 작음 (목표 해상도의 1/4 이하)
3. 훈련 데이터와 고해상도 데이터의 물리적 도메인이 다름

**확인 사항**:
```python
# Normalizer output_mode 확인
print(f"Normalizer mode: {normalizer.output_mode}")

# n_modes vs 해상도 비율 확인
print(f"n_modes: {model.n_modes}")
print(f"Input resolution: {x_hr.shape[2:]}")
print(f"Ratio: {x_hr.shape[2] / model.n_modes[0]:.2f}")  # > 2 권장
```

## 참고 자료

- FNO 논문: [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)
- TFNO 논문: [Multi-Grid Tensorized Fourier Neural Operator](https://openreview.net/forum?id=AWiDlO63bH)
- NeuralOperator 라이브러리: [neuraloperator/neuraloperator](https://github.com/neuraloperator/neuraloperator)
