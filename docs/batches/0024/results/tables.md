| cell | NMSE of the eight nulls | largest / smallest null (N1: at most 3) | largest variant claimed the same / median null (N2: at most 4) | smallest variant claimed different / largest null (N3: at least 1.5) | broken / largest null | nulls |
|---|---|---|---|---|---|---|
| llama.cpp, Q8_0 weights | 0.000559 to 0.000791 | 1.41 | 0.95 | 2.20 | 3.52e+03 | 8 |
| llama.cpp, Q4_0 weights | 0.000567 to 0.000863 | 1.52 | 1.31 | 49.12 | 3.25e+03 | 8 |
| llama.cpp, f16 cache | 1.14e-07 to 1.56e-07 | 1.37 | 1.18 |  | 1.79e+07 | 8 |
| llama.cpp, Q8_0 cache, rotated | 2.43e-05 to 3.49e-05 | 1.43 | 1.19 | 4.02 | 8.02e+04 | 8 |
| llama.cpp, Q4_0 cache, rotated | 0.00116 to 0.00152 | 1.31 | 1.58 | 26.00 | 1.88e+03 | 8 |
| llama.cpp, Q8_0 cache, not rotated | 4.19e-05 to 6.24e-05 | 1.49 | 1.21 | 5.79 | 4.48e+04 | 8 |
| llama.cpp, Q4_0 cache, not rotated | 0.00112 to 0.00179 | 1.60 | 1.54 | 22.48 | 1.63e+03 | 8 |
| CTranslate2, int8 | 0.00567 to 0.00711 | 1.25 | 83.21 |  | 403 | 8 |

#### llama.cpp, Q8_0 weights

The eight nulls: 0.000559 to 0.000791, median 0.00073.

| variant | class claimed | NMSE | in units of the median null | in units of the largest null |
|---|---|---|---|---|
| heads | same | 0.000428 | 0.59 | 0.54 |
| resid_blk | same | 0.000527 | 0.72 | 0.67 |
| units_blk | same | 0.000693 | 0.95 | 0.88 |
| units | different | 0.00174 | 2.38 | 2.20 |
| resid | different | 0.00219 | 3.00 | 2.77 |
| canon | different | 0.0031 | 4.25 | 3.92 |
| perm | different | 0.00316 | 4.33 | 4.00 |
| scale | different | 0.0545 | 74.65 | 68.93 |
| all | different | 0.0581 | 79.62 | 73.52 |
| broken | control (broken) | 2.78 | 3809.97 | 3518.03 |

#### llama.cpp, Q4_0 weights

The eight nulls: 0.000567 to 0.000863, median 0.000794.

| variant | class claimed | NMSE | in units of the median null | in units of the largest null |
|---|---|---|---|---|
| heads | same | 0.000402 | 0.51 | 0.47 |
| units_blk | same | 0.00098 | 1.23 | 1.14 |
| resid_blk | same | 0.00104 | 1.31 | 1.20 |
| units | different | 0.0424 | 53.37 | 49.12 |
| canon | different | 0.12 | 151.00 | 138.98 |
| resid | different | 0.123 | 155.20 | 142.85 |
| perm | different | 0.152 | 191.61 | 176.35 |
| scale | different | 1.06 | 1331.80 | 1225.79 |
| all | different | 1.4 | 1769.10 | 1628.28 |
| broken | control (broken) | 2.8 | 3525.83 | 3245.17 |

#### llama.cpp, f16 cache

The eight nulls: 1.14e-07 to 1.56e-07, median 1.27e-07.

| variant | class claimed | NMSE | in units of the median null | in units of the largest null |
|---|---|---|---|---|
| heads | same | 1.03e-07 | 0.81 | 0.66 |
| scale | same | 1.07e-07 | 0.84 | 0.69 |
| units | same | 1.21e-07 | 0.96 | 0.78 |
| all | same | 1.25e-07 | 0.99 | 0.80 |
| units_blk | same | 1.3e-07 | 1.03 | 0.84 |
| perm | same | 1.31e-07 | 1.03 | 0.84 |
| resid | same | 1.38e-07 | 1.09 | 0.89 |
| canon | same | 1.49e-07 | 1.17 | 0.95 |
| resid_blk | same | 1.49e-07 | 1.18 | 0.96 |
| broken | control (broken) | 2.79 | 22051656.19 | 17933026.77 |

#### llama.cpp, Q8_0 cache, rotated

The eight nulls: 2.43e-05 to 3.49e-05, median 2.89e-05.

| variant | class claimed | NMSE | in units of the median null | in units of the largest null |
|---|---|---|---|---|
| units | same | 2.77e-05 | 0.96 | 0.80 |
| resid_blk | same | 2.89e-05 | 1.00 | 0.83 |
| units_blk | same | 3.05e-05 | 1.06 | 0.88 |
| heads | same | 3.08e-05 | 1.06 | 0.88 |
| perm | same | 3.32e-05 | 1.15 | 0.95 |
| resid | same | 3.44e-05 | 1.19 | 0.99 |
| canon | different | 0.00014 | 4.85 | 4.02 |
| all | different | 0.00619 | 214.02 | 177.57 |
| scale | different | 0.00922 | 318.57 | 264.32 |
| broken | control (broken) | 2.8 | 96619.69 | 80166.01 |

#### llama.cpp, Q4_0 cache, rotated

The eight nulls: 0.00116 to 0.00152, median 0.00136.

| variant | class claimed | NMSE | in units of the median null | in units of the largest null |
|---|---|---|---|---|
| resid | same | 0.00115 | 0.85 | 0.76 |
| heads | same | 0.00142 | 1.05 | 0.94 |
| units | same | 0.00153 | 1.13 | 1.01 |
| units_blk | same | 0.00181 | 1.33 | 1.19 |
| resid_blk | same | 0.00196 | 1.44 | 1.29 |
| perm | same | 0.00214 | 1.58 | 1.41 |
| canon | different | 0.0395 | 29.13 | 26.00 |
| all | different | 1.23 | 909.95 | 812.21 |
| scale | different | 1.31 | 969.48 | 865.35 |
| broken | control (broken) | 2.86 | 2105.69 | 1879.52 |

#### llama.cpp, Q8_0 cache, not rotated

The eight nulls: 4.19e-05 to 6.24e-05, median 5.02e-05.

| variant | class claimed | NMSE | in units of the median null | in units of the largest null |
|---|---|---|---|---|
| heads | same | 4.05e-05 | 0.81 | 0.65 |
| units_blk | same | 4.79e-05 | 0.95 | 0.77 |
| resid | same | 4.85e-05 | 0.97 | 0.78 |
| resid_blk | same | 5.05e-05 | 1.01 | 0.81 |
| perm | same | 5.55e-05 | 1.11 | 0.89 |
| units | same | 6.1e-05 | 1.21 | 0.98 |
| canon | different | 0.000361 | 7.19 | 5.79 |
| all | different | 0.0297 | 590.71 | 475.40 |
| scale | different | 0.0431 | 859.30 | 691.56 |
| broken | control (broken) | 2.8 | 55700.62 | 44827.50 |

#### llama.cpp, Q4_0 cache, not rotated

The eight nulls: 0.00112 to 0.00179, median 0.0015.

| variant | class claimed | NMSE | in units of the median null | in units of the largest null |
|---|---|---|---|---|
| units | same | 0.00139 | 0.93 | 0.78 |
| units_blk | same | 0.0015 | 1.00 | 0.84 |
| resid_blk | same | 0.00184 | 1.22 | 1.03 |
| heads | same | 0.00201 | 1.34 | 1.12 |
| resid | same | 0.0021 | 1.40 | 1.17 |
| perm | same | 0.00231 | 1.54 | 1.29 |
| canon | different | 0.0402 | 26.80 | 22.48 |
| all | different | 1.28 | 853.14 | 715.52 |
| scale | different | 1.36 | 904.22 | 758.36 |
| broken | control (broken) | 2.91 | 1939.06 | 1626.27 |

#### CTranslate2, int8

The eight nulls: 0.00567 to 0.00711, median 0.00635.

| variant | class claimed | NMSE | in units of the median null | in units of the largest null |
|---|---|---|---|---|
| heads | same | 0 | 0.00 | 0.00 |
| units | same | 0 | 0.00 | 0.00 |
| resid_blk | same | 0.00423 | 0.67 | 0.60 |
| perm | same | 0.005 | 0.79 | 0.70 |
| resid | same | 0.005 | 0.79 | 0.70 |
| canon | same | 0.0271 | 4.26 | 3.81 |
| all | same | 0.396 | 62.34 | 55.75 |
| units_blk | same | 0.416 | 65.53 | 58.61 |
| scale | same | 0.529 | 83.21 | 74.42 |
| broken | control (broken) | 2.87 | 451.10 | 403.42 |

