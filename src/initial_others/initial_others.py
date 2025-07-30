import numpy as np
import pandas as pd
from scipy.stats import qmc

# 샘플링 파라미터 설정
pressure_min = 0.001
pressure_max = 0.01
num_samples = 10  # 샘플 개수

# uniform sampling
sampler = qmc.LatinHypercube(d=2)
sample = sampler.random(n=num_samples)

# pressure: scale to [pressure_min, pressure_max], then to actual pressure
pressures = 501325 + 9759.14 * (pressure_min + (pressure_max - pressure_min) * sample[:, 0])
# ratio: scale to [0, 1]
ratios = sample[:, 1]

# DataFrame 생성
df = pd.DataFrame({
    'pressure': pressures,
    'ratio': ratios
})

# CSV로 저장
df.to_csv('./src/initial_others/output/pressure_ratio_samples.csv', index=False)