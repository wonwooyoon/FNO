# FNO_pure.py CSV 저장 기능 구현 완료

## 🎯 구현된 기능

FNO_pure.py에 **SAVEASCSV** 옵션을 추가하여 visualization 함수에서 생성되는 모든 이미지 데이터를 CSV 형태로 저장할 수 있는 기능을 구현했습니다.

## 📋 구현 내용

### 1. CONFIG 설정 추가
```python
'VISUALIZATION': {
    'SAMPLE_NUM': 8,
    'TIME_INDICES': (4, 9, 14, 19),
    'DPI': 200,
    'SAVEASCSV': False  # CSV 저장 옵션 추가
},
```

### 2. CSV 데이터 구조
- **좌표 시스템**: (x_coord, y_coord)로 각 픽셀의 위치 정보
- **데이터 형식**: `sample_time_type` 패턴 (예: `8_4_gt`, `8_4_pred`, `8_4_error`)
- **Time indices**: (4, 9, 14, 19) 시점의 데이터
- **데이터 타입**: ground truth, prediction, error 3가지

### 3. CSV 파일 구조
```
x_coord | y_coord | 8_4_gt | 8_4_pred | 8_4_error | 8_9_gt | 8_9_pred | 8_9_error | ... 
   0    |    0    | val1   |   val2   |    val3   | val4   |   val5   |    val6   | ...
   0    |    1    | val7   |   val8   |    val9   | val10  |   val11  |   val12   | ...
```

## 🔍 테스트 결과

### CSV 파일 정보
- **파일명**: `FNO_visualization_data.csv`
- **저장위치**: 기존 이미지 저장 디렉토리와 동일 (`OUTPUT_DIR`)
- **크기**: (2048, 14) - 64×32 픽셀 = 2048 행, 14개 열
- **컬럼수**: 2개 좌표 + 4개 시점 × 3개 타입 = 14개 컬럼

### 컬럼 구성
1. `x_coord`, `y_coord` - 공간 좌표
2. `8_4_gt`, `8_4_pred`, `8_4_error` - 시점 4에서의 데이터
3. `8_9_gt`, `8_9_pred`, `8_9_error` - 시점 9에서의 데이터
4. `8_14_gt`, `8_14_pred`, `8_14_error` - 시점 14에서의 데이터
5. `8_19_gt`, `8_19_pred`, `8_19_error` - 시점 19에서의 데이터

## ✅ 검증 완료

### 기능 테스트
- ✅ **CONFIG 옵션**: `SAVEASCSV=True`로 설정 시 CSV 파일 생성
- ✅ **데이터 정확성**: ground truth, prediction, error 값이 올바르게 저장
- ✅ **파일 구조**: 예상한 대로 (x, y, 이미지출력값) 형태의 컬럼 구조
- ✅ **저장 위치**: 기존 이미지 저장 위치와 동일한 디렉토리
- ✅ **성능**: 훈련 및 시각화 과정에 영향 없이 CSV 생성

### 샘플 데이터 확인
```csv
x_coord,y_coord,8_4_gt,8_4_pred,8_4_error,...
0,0,2.111922e-12,-1.2834803e-08,1.28369155e-08,...
0,1,2.5368785e-12,-1.1499239e-08,1.15017755e-08,...
```

## 🚀 사용 방법

1. **CONFIG 설정**: `CONFIG['VISUALIZATION']['SAVEASCSV'] = True`
2. **모델 훈련**: 기존과 동일하게 FNO_pure.py 실행
3. **결과 확인**: `OUTPUT_DIR/FNO_visualization_data.csv` 파일 생성 확인

## 📊 활용 가능성

- **데이터 분석**: Python/R로 추가 분석 가능
- **외부 도구 연동**: Excel, Tableau 등에서 활용
- **비교 연구**: 다른 모델과의 정량적 비교 용이
- **후처리**: 커스텀 시각화 및 통계 분석

**✨ FNO_pure.py에 CSV 저장 기능이 성공적으로 구현되었습니다!**