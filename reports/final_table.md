# Final Benchmarking Master Table

**Evaluation Protocol:** Strict Chronological 70/10/20 Partition, Train-Only MinMaxScaler, Boundary-Aware Slicing ($W=15$, $N=5,081$ test sequences).
**Hardware Platform:** Windows-11-10.0.26300-SP0 | CPU (oneDNN enabled)
**Statistical Variance:** Evaluated across 5 random seeds (Bootstrap $B=5$, mean $\pm$ std).

| Model ID | Architecture | Params | Latency (ms) | pH MAE | pH RMSE | pH $R^2$ | TDS MAE (ppm) | TDS RMSE (ppm) | TDS $R^2$ | Avg $R^2$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0** | Persistence Baseline | 0 | 0.01 | 0.0420 | 0.0548 | -0.2804 | 0.48 | 3.15 | 0.9933 | 0.7409 |
| **B1** | CNN-BiLSTM (Baseline) | 74,153 | 56.16 | 0.0442 | 0.0533 | -0.2108 | 53.44 | 54.22 | -0.9930 | 0.1058 |
| **B2** | Vanilla LSTM | 20,165 | 49.37 | 0.0366 | 0.0445 | 0.1544 | 22.03 | 24.51 | 0.5927 | 0.5324 |
| **B3** | Gated Recurrent Unit | 15,877 | 65.83 | 0.0677 | 0.0823 | -1.8873 | 73.72 | 74.80 | -2.7935 | -0.7424 |
| **B4** | Bidirectional LSTM | 25,797 | 92.51 | 0.0603 | 0.0729 | -1.2671 | 143.00 | 147.52 | -13.7545 | -3.3965 |
| **B5** | Temporal ConvNet | 65,413 | 29.41 | 0.0392 | 0.0478 | 0.0248 | 20.15 | 22.87 | 0.6453 | 0.4729 |
| **B6** | Transformer Encoder | 53,061 | 35.49 | 0.0781 | 0.0871 | -2.2339 | 83.55 | 85.88 | -4.0002 | -1.0542 |
| **B7** | CNN + GRU Hybrid | 41,541 | 37.43 | 0.0424 | 0.0526 | -0.1789 | 140.60 | 149.48 | -14.1479 | -2.7939 |
| **MT-TCN-LSTM** | MT-TCN-LSTM (Proposed) | 75,846 | 67.70 / 3990.1 (MC) | 0.0416 | 0.0516 | -0.1331 | 131.65 | 132.95 | -10.9833 | -2.1766 |


### 5-Seed Bootstrap Uncertainty (Mean ± Std)

| Model ID | Architecture | pH MAE (5-Seed CI) | TDS MAE ppm (5-Seed CI) |
| :--- | :--- | :---: | :---: |
| **B0** | Persistence Baseline | 0.0418 ± 0.0006 | 0.49 ± 0.06 |
| **B1** | CNN-BiLSTM (Baseline) | 0.0443 ± 0.0003 | 53.34 ± 0.10 |
| **B2** | Vanilla LSTM | 0.0365 ± 0.0004 | 22.00 ± 0.14 |
| **B3** | Gated Recurrent Unit | 0.0674 ± 0.0005 | 73.58 ± 0.24 |
| **B4** | Bidirectional LSTM | 0.0602 ± 0.0006 | 142.97 ± 0.56 |
| **B5** | Temporal ConvNet | 0.0392 ± 0.0004 | 20.05 ± 0.20 |
| **B6** | Transformer Encoder | 0.0780 ± 0.0007 | 83.53 ± 0.26 |
| **B7** | CNN + GRU Hybrid | 0.0424 ± 0.0005 | 140.77 ± 0.99 |
| **MT-TCN-LSTM** | MT-TCN-LSTM (Proposed) | 0.0416 ± 0.0003 | 131.40 ± 0.21 |
