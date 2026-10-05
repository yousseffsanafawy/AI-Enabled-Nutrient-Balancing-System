# Sprint 11: Final Paper Evaluation & Conference Write-Up

## Executive Summary
In **Sprint 11**, we executed the final out-of-sample evaluation of the proposed **MT-TCN-LSTM** architecture on the held-out chronological test partition (5,081 sequences, rows 20,456 to 25,569 of `IoTData_25K_without_interpolation.csv`). We synthesized the findings into a camera-ready **5-page conference paper / edited chapter draft** adhering strictly to the **JACK Conference Empirical Evaluation Track** and the **Humanizer Skill** (`SKILL_Humanaized.md`).

---

## 1. Final Test Partition Evaluation (`exp_011`)

The model was evaluated once on the test partition without hyperparameter retuning:

| Modality / Task | Final Test Metric | Physical Target / Deadband | Baseline CNN-BiLSTM | Statistical Improvement |
|---|:---:|:---:|:---:|:---:|
| **pH Forecasting** | **0.0416 MAE** (0.0516 RMSE) | Horticultural Range 5.5–6.5 | 0.0436 MAE | $+4.6\%$ error reduction ($p < 0.001$) |
| **TDS Forecasting** | **131.65 ppm MAE** | Vegetative Target 650 ± 20 ppm | 141.52 ppm MAE | $+7.0\%$ error reduction ($p < 0.001$) |
| **Air Temperature** | **0.0543 °C MAE** | Safe Range 18.0–26.0 °C | 0.0612 °C MAE | $+11.3\%$ error reduction |
| **Relative Humidity**| **0.1874 % MAE** | VPD Optimization 60–70% | 0.2140 % MAE | $+12.4\%$ error reduction |
| **Single-Sample Latency** | **71.69 ms** | Real-Time Hardware Budget (10.0s) | 88.60 ms | $19.1\%$ faster execution |
| **Quantized INT8 Footprint** | **112.5 KB** | Flash Budget 4 MB / Heap 160 KB | 280.4 KB | $64.8\%$ storage reduction |

Experiment log archived: [`experiments/exp_011_final_paper_evaluation.json`](file:///c:/Users/youse/Downloads/JackHom/REPO/AI-Enabled-Nutrient-Balancing-System/experiments/exp_011_final_paper_evaluation.json).

---

## 2. Five-Page Conference Paper Structure & Density

The conference paper has been tailored for the strict 5-page limit of the conference evaluation track:

* **Publication Markdown Draft:** [`reports/paper_draft.md`](file:///c:/Users/youse/Downloads/JackHom/REPO/AI-Enabled-Nutrient-Balancing-System/reports/paper_draft.md)
* **Camera-Ready IEEE LaTeX Source:** [`reports/paper_draft.tex`](file:///c:/Users/youse/Downloads/JackHom/REPO/AI-Enabled-Nutrient-Balancing-System/reports/paper_draft.tex)

### Page-by-Page Allocation

```
Page 1: Title, Abstract, Section I (Introduction, The 10-Second Physical Reality, Autocorrelation Dilemma, 5 Contributions)
Page 2: Section II (Dataset Realities & Forensics, Table I: Split Protocol Leakage Illusion, Boundary-Aware Preprocessing)
Page 3: Section III (Proposed MT-TCN-LSTM Architecture Box, Multi-Task Loss, Epistemic MC Dropout, 5-Tier Safety Engine)
Page 4: Section IV (Table II: Benchmark Results Across 9 Models, Table III: Systematic Architectural Ablations)
Page 5: Section V, VI & References (Table IV: 10 Physical Titrations, ESP32 Heap Constraints, Edge Gateway, Agronomic Attribution, References)
```

---

## 3. Humanizer Skill Compliance Audit

Under the directives of [`SKILL_Humanaized.md`](file:///c:/Users/youse/Downloads/JackHom/REPO/AI-Enabled-Nutrient-Balancing-System/SKILL_Humanaized.md):

1. **AI Banned Words Count: Exactly 0**
   - Verified absence of: `delve`, `landscape`, `crucial`, `vital`, `pivotal`, `leverage`, `furthermore`, `moreover`, `in addition`, `navigate`, `robust`, `comprehensive`, `holistic`, `foster`, `facilitate`, `ensure`.
2. **Hedging Chains Count: Exactly 0**
   - Verified absence of: `It is important to note`, `It is worth mentioning`, `One might argue`, `Needless to say`.
3. **Prose Word Count: 2,533 words**
   - Fits the 5-page conference limit without overflow.
4. **Tone & Stance:** Authentic peer-to-peer engineering review exposing data leakage, naive persistence baselines, and microcontroller heap memory limitations.
