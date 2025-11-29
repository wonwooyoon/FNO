# Preprocessing Pipeline

간결하고 직관적인 PFLOTRAN 데이터 전처리 파이프라인

## 파일 구조

```
src/preprocessing/
├── preprocessing_collect.py      # 데이터 수집 및 병합
├── preprocessing_normalize.py    # Normalization 메인 로직
├── normalizer_core.py            # 핵심 Normalizer 클래스
├── normalizer_utils.py           # 통계/시각화 유틸리티
└── README.md                     # 이 파일
```

## 사용 방법

### 1단계: 데이터 수집 및 병합 (preprocessing_collect.py)

로컬 및 원격 서버에서 PFLOTRAN HDF5 데이터를 수집하여 하나의 PyTorch tensor 파일로 병합합니다.

#### LR (Low-Resolution) 모드
```bash
# 원격 서버 포함
python preprocessing_collect.py --mode lr --config servers.yaml

# 로컬 데이터만
python preprocessing_collect.py --mode lr --local-only
```

**출력:** `src/preprocessing/merged_raw.pt` (N, 11, nx, ny, nt)

#### HR (High-Resolution) 모드
```bash
python preprocessing_collect.py --mode hr --config servers.yaml
```

**출력:** `src/preprocessing/merged_raw_hr.pt` (N, 11, nx_hr, ny_hr, nt)

### 2단계: Normalization (preprocessing_normalize.py)

채널별 transformation과 normalization을 적용합니다.

#### LR 모드 (새 normalizer 생성)
```bash
# 상대 경로 사용 (권장: 자동으로 src/preprocessing/ 기준)
python preprocessing_normalize.py \
    --mode lr \
    --input merged_raw.pt \
    --output-mode log \
    --output merged_normalized.pt
```

**Output modes:**
- `log`: log10 transformation (기본, 추천)
- `raw`: raw 값 사용 (linear scale)
- `delta`: t=0 대비 변화량

**출력:**
- `src/preprocessing/merged_normalized.pt`: 정규화된 데이터
- `src/preprocessing/normalizer_log.pkl`: Normalizer 객체
- `src/preprocessing/normalization_stats/`: 통계 분석 결과

#### HR 모드 (기존 normalizer 재사용)
```bash
# 상대 경로 사용 (권장: 자동으로 src/preprocessing/ 기준)
python preprocessing_normalize.py \
    --mode hr \
    --input merged_raw_hr.pt \
    --normalizer normalizer_log.pkl \
    --output merged_normalized_hr.pt
```

**출력:**
- `src/preprocessing/merged_normalized_hr.pt`: 정규화된 HR 데이터

⚠️ **중요:** HR 데이터는 LR normalizer를 재사용해야 합니다 (zero-shot super-resolution)

## 채널 설정

### 입력 채널 (11개)

| Channel | Name               | Transform       | Normalizer    | Description          |
|---------|--------------------|-----------------|---------------|----------------------|
| 0       | Perm               | log10           | UnitGaussian  | Permeability         |
| 1       | Calcite            | shifted_log (ε=1e-6) | UnitGaussian  | Calcite VF           |
| 2       | Clino              | shifted_log (ε=1e-6) | UnitGaussian  | Clinochlore VF       |
| 3       | Pyrite             | shifted_log (ε=1e-9) | UnitGaussian  | Pyrite VF            |
| 4       | Smectite           | none            | UnitGaussian  | Smectite VF          |
| 5       | Material_Source    | none            | none          | Source (one-hot)     |
| 6       | Material_Bentonite | none            | none          | Bentonite (one-hot)  |
| 7       | Material_Fracture  | none            | none          | Fracture (one-hot)   |
| 8       | Vx                 | none            | UnitGaussian  | X-Velocity           |
| 9       | Vy                 | none            | UnitGaussian  | Y-Velocity           |
| 10      | Meta               | none            | UnitGaussian  | Meta parameter       |

### 출력 채널 (1개)

| Mode  | Transform | Description                  |
|-------|-----------|------------------------------|
| log   | log10     | log10(U) - 기본, 추천        |
| raw   | none      | 원본 농도값                  |
| delta | delta     | t=0 대비 변화량 (source mask)|

## 전체 워크플로우

### Low-Resolution 데이터

```bash
# 1. 데이터 수집
python preprocessing_collect.py --mode lr --config servers.yaml
# → src/preprocessing/merged_raw.pt

# 2. Normalization (상대 경로 사용)
python preprocessing_normalize.py \
    --mode lr \
    --input merged_raw.pt \
    --output-mode log \
    --output merged_normalized.pt
# → src/preprocessing/merged_normalized.pt
# → src/preprocessing/normalizer_log.pkl (중요!)
# → src/preprocessing/normalization_stats/

# 3. FNO 학습
python src/FNO/FNO.py  # normalizer_log.pkl 사용
```

### High-Resolution 데이터 (Zero-shot Super-Resolution)

```bash
# 1. HR 데이터 수집
python preprocessing_collect.py --mode hr --config servers.yaml
# → src/preprocessing/merged_raw_hr.pt

# 2. LR normalizer로 정규화 (상대 경로 사용)
python preprocessing_normalize.py \
    --mode hr \
    --input merged_raw_hr.pt \
    --normalizer normalizer_log.pkl \
    --output merged_normalized_hr.pt
# → src/preprocessing/merged_normalized_hr.pt

# 3. FNO 평가 (LR에서 학습한 모델 사용)
python src/FNO/FNO_eval.py --test-data merged_normalized_hr.pt
```

## 주요 옵션

### preprocessing_collect.py

```
--mode {lr,hr}          처리 모드 (필수)
--config PATH           서버 설정 YAML 파일
--local-only            로컬 데이터만 처리
```

### preprocessing_normalize.py

```
--mode {lr,hr}          처리 모드 (필수)
--input PATH            입력 raw 데이터 (필수)
--output PATH           출력 normalized 데이터 (필수)

LR 모드 전용:
--output-mode {log,raw,delta}  출력 transformation 모드 (필수)
--no-analysis           통계 분석 생략
--analysis-samples N    분석용 샘플 개수 (기본: 3000)

HR 모드 전용:
--normalizer PATH       LR normalizer pickle 파일 (필수)
```

**경로 처리 규칙:**
- 상대 경로 입력 시: `src/preprocessing/` 기준으로 저장
- 절대 경로 입력 시: 지정한 경로에 저장
- Normalizer와 통계는 항상 `src/preprocessing/`에 저장

## 서버 설정 예시 (servers.yaml)

```yaml
servers:
  - host: server1.example.com
    user: username
    port: 22

  - host: server2.example.com
    user: username
    port: 2222
```

## 데이터 형식

### Raw 데이터 (merged_raw.pt)
```python
{
    'x': Tensor (N, 11, nx, ny, nt),     # Raw input with t=0
    'y': Tensor (N, 1, nx, ny, nt),      # Raw output with t=0
    'xc': Tensor (nx,),                   # X coordinates
    'yc': Tensor (ny,),                   # Y coordinates
    'time_keys': List[str]                # Time labels
}
```

### Normalized 데이터 (merged_normalized.pt)
```python
{
    'x': Tensor (N, 11, nx, ny, nt-1),   # Normalized input (t=0 제거)
    'y': Tensor (N, 1, nx, ny, nt-1),    # Normalized output (t=0 제거)
    'xc': Tensor (nx,),
    'yc': Tensor (ny,),
    'time_keys': List[str]
}
```

## 트러블슈팅

### ImportError: No module named 'neuraloperator'
```bash
cd neuraloperator
pip install -e .
```

### SSH 연결 실패
- `servers.yaml`의 호스트/포트 확인
- SSH 키 설정 확인: `ssh-copy-id user@host`

### Memory Error
- `--analysis-samples` 옵션으로 샘플 수 줄이기
- 서버별로 데이터 분산 처리

### 채널 수 불일치
- Input은 항상 11채널 (10 + meta)
- Output은 항상 1채널 (Uranium)
- 기존 데이터 재확인 필요

## 참고 사항

1. **t=0 처리**
   - Raw 데이터에는 t=0 포함 (nt개 타임스텝)
   - Normalized 데이터는 t=0 제거 (nt-1개 타임스텝)
   - Delta mode는 t=0을 reference로 사용 후 제거

2. **HR Normalization**
   - 반드시 LR normalizer 재사용
   - 새로운 normalizer 생성하면 안 됨
   - 해상도 차이는 FNO가 자동 처리

3. **Output Mode 선택**
   - `log`: 대부분의 경우 권장 (농도값 range가 넓음)
   - `raw`: 선형 관계 학습 시
   - `delta`: 변화량 학습 시 (초기 조건 독립적)

4. **통계 분석**
   - `normalization_stats/` 폴더에 자동 생성
   - Histogram, CSV 포함
   - 정규화 품질 확인용
