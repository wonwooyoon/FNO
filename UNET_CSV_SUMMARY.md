# UNet_pure.py CSV 저장 기능 구현 완료

## 🎯 구현 완료

**UNet_pure.py에도 FNO_pure.py와 동일한 CSV 저장 기능을 성공적으로 추가했습니다!**

## 📋 구현된 기능

### 1. CONFIG 설정 추가
```python
'VISUALIZATION': {
    'SAMPLE_NUM': 8,
    'TIME_INDICES': (3, 7, 11, 15),  # UNet 고유 시점
    'DPI': 200,
    'SAVEASCSV': False  # CSV 저장 옵션 추가
},
```

### 2. Import 추가
```python
import pandas as pd
```

### 3. CSV 저장 기능
- FNO_pure.py와 동일한 로직 구현
- UNet에 맞는 파일명으로 수정: `UNet_visualization_data.csv`
- UNet 고유의 시간 인덱스 사용: (3, 7, 11, 15)

## 🔍 테스트 결과

### CSV 파일 정보
- **파일명**: `UNet_visualization_data.csv`
- **저장위치**: `OUTPUT_DIR` (기존 이미지와 동일)
- **크기**: (2048, 14) - 64×32 픽셀 = 2048행, 14개 열
- **컬럼 구성**: 2개 좌표 + 4개 시점 × 3개 타입 = 14개 컬럼

### 컬럼 명세
```
['x_coord', 'y_coord', 
 '8_3_gt', '8_3_pred', '8_3_error',    # 시점 3
 '8_7_gt', '8_7_pred', '8_7_error',    # 시점 7  
 '8_11_gt', '8_11_pred', '8_11_error', # 시점 11
 '8_15_gt', '8_15_pred', '8_15_error'] # 시점 15
```

## 📊 FNO vs UNet CSV 비교

| 특성 | FNO_pure.py | UNet_pure.py |
|------|-------------|--------------|
| **파일명** | `FNO_visualization_data.csv` | `UNet_visualization_data.csv` |
| **시간 인덱스** | (4, 9, 14, 19) | (3, 7, 11, 15) |
| **데이터 구조** | 동일 (2048, 14) | 동일 (2048, 14) |
| **컬럼 패턴** | `8_4_gt`, `8_9_pred`... | `8_3_gt`, `8_7_pred`... |

## ✅ 검증 완료

### 테스트 통과
- ✅ **CONFIG 옵션**: `SAVEASCSV=True` 설정 시 CSV 생성
- ✅ **데이터 정확성**: ground truth, prediction, error 값 정확히 저장
- ✅ **파일 구조**: (x, y, 값) 형태의 올바른 컬럼 구조
- ✅ **시간 인덱스**: UNet 고유 시점 (3, 7, 11, 15) 정확히 반영
- ✅ **성능**: 훈련 및 시각화에 영향 없이 CSV 생성

### 샘플 데이터
```csv
x_coord,y_coord,8_3_gt,8_3_pred,8_3_error,...
0,0,1.526795e-12,6.5477614e-09,-6.5462347e-09,...
0,1,1.8357974e-12,8.41348e-09,-8.411644e-09,...
```

## 🚀 사용 방법

```python
# UNet_pure.py에서 CSV 저장 활성화
CONFIG['VISUALIZATION']['SAVEASCSV'] = True
```

## 🎉 완료 사항

이제 **FNO_pure.py**와 **UNet_pure.py** 모두에서:
1. ✅ **3-way 데이터 분할** (train/validation/test)
2. ✅ **듀얼 손실 메트릭** (L2 + MSE)
3. ✅ **모드별 실행** (single/optuna/eval)
4. ✅ **CSV 데이터 저장** (시각화 데이터 추출)
5. ✅ **이모지 제거** (깔끔한 출력)

모든 기능이 완벽하게 구현되어 두 모델을 동일한 방식으로 사용할 수 있습니다!