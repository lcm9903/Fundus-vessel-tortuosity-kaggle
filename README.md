# 안저 영상 혈관 분할 및 굴곡도 정량화 (FIVES)

U-Net으로 안저 영상의 망막 혈관을 분할하고, 혈관 중심선에서 세그먼트별 **굴곡도(tortuosity)** 를 정량화하는 Kaggle 노트북입니다. FIVES 데이터로 분할 모델을 학습·평가한 뒤, **모델이 잰 굴곡도를 전문가 정답 기준 굴곡도와 비교해 얼마나 믿을 수 있는지** 확인하고, 질환군(정상·AMD·DR·녹내장) 간 굴곡도를 비교합니다.

- 노트북: `fundus-vessel-tortuosity-kaggle.ipynb` (1차: 학습 + FIVES 분석)
- 후속 노트북: `aptos_tortuosity_dr_analysis_kaggle.ipynb` (2차: 학습된 모델로 APTOS 2019의 DR 등급–굴곡도 상관관계 분석)
- 실행 환경: Kaggle Notebook, GPU T4 ×2

---

## 1. 연구 질문

1. FIVES로 학습한 U-Net은 굴곡도 계산에 필요할 만큼 혈관을 **연결성 있게** 분할하는가?
2. 예측 마스크로 잰 굴곡도는 정답 마스크로 잰 굴곡도와 얼마나 일치하는가? 어떤 지표가 가장 믿을 만한가?
3. 분할 오차가 **특정 질환군(특히 DR)에서만 굴곡도를 부풀리는가?**
4. 분할 오류가 섞이지 않은 정답 마스크 기준으로, 질환군 간 굴곡도 차이가 있는가?

## 2. 결과 요약

| 항목 | 결과 |
|---|---|
| 분할 성능 (FIVES test 200장) | **Dice 0.907**, clDice 0.916, AUC 0.994 |
| 가장 신뢰할 수 있는 굴곡도 지표 | **SOAM** (정답–예측 ICC 0.79) |
| 신뢰도가 낮은 지표 | DM(ICC 0.43), ICM(ICC 0.40) |
| DR군에서만 굴곡도가 부풀려지는가 | 아니요. 예측 굴곡도는 전반적으로 약간 낮게 나오며, 그 정도가 DR군에서 더 크지 않음 |
| 정답 기준 DR vs 정상 | DM·ICM·TD는 차이 없음. SOAM·CE는 DR군이 약간 **낮음**. 혈관 밀도는 DR군이 뚜렷이 낮음 |

## 3. 파이프라인

```
FIVES 로드 ─► 전처리(FOV 정규화, CLAHE) ─► U-Net 학습 ─► 분할 평가
   ─► 스켈레톤화 ─► 세그먼트 분리 ─► 세그먼트별 굴곡도 ─► 영상 단위 집계
   ─► 정답/예측 굴곡도 일치도(편향 점검) ─► 정답 기준 질환군 비교
```

| 절 | 내용 |
|---|---|
| 1 | FIVES 자동 탐색(없으면 kagglehub 다운로드), FOV 기준 크롭·크기 정규화, CLAHE, 품질 지표 |
| 2 | U-Net(ResNet34) 학습: BCE + Dice + soft clDice, 검증셋으로 임계값 선택 |
| 3 | FIVES 테스트셋 분할 평가 (전체 및 질환군별) |
| 4 | 스켈레톤 → 세그먼트 분리 → 굴곡도 지표 5종, 합성 곡선으로 계산 검증 |
| 5 | 정답/예측 굴곡도 일치도와 질환군별 편향, 정답 기준 질환군 비교 |

## 4. 데이터

**FIVES** (Fundus Image dataset for Vessel Segmentation)
- 800장, 2048×2048, 전문가가 픽셀 단위로 칠한 혈관 정답 포함
- 정상·AMD·DR·녹내장 각 200장, 공식 분할 train 600 / test 200 (각 질환 150 / 50)
- 파일명 형식: `번호_질환코드.png` (A=AMD, D=DR, G=녹내장, N=정상)
- Kaggle 사본 `nikitamanaenkov/fundus-image-dataset-for-vessel-segmentation` 사용
- train 600장 중 질환별 층화로 10%(60장)를 검증셋으로 분리 → **train 540 / val 60 / test 200**

## 5. 방법

### 5.1 전처리
- FOV(촬영 원형 영역) 검출 → 경계 상자로 크롭 → 정사각형 패딩 → **FOV 지름을 1024px로 정규화**
  - 해상도와 카메라가 다른 데이터셋 사이에서도 길이·곡률(px 단위)을 비교할 수 있게 하기 위한 단계입니다.
- LAB 색공간 L 채널에 CLAHE(clipLimit 2.0, 8×8)를 적용하고, FOV 밖은 0으로 채웁니다.
- 흐림 정도(라플라시안 분산), 밝기, 대비, FOV 비율·종횡비를 품질 지표로 기록합니다.
- FIVES의 FOV 내 혈관 픽셀 비율은 9.7%입니다.

### 5.2 분할 모델
| 항목 | 설정 |
|---|---|
| 모델 | `segmentation_models_pytorch` U-Net, ResNet34 인코더(ImageNet 사전학습) |
| 입력 | FOV 안쪽 중심 512×512 무작위 패치 (에폭당 1,200개), 추론은 1024 전체 영상 |
| 손실 | BCE + Dice + 0.5 × soft clDice |
| 증강 | 반전, 90° 회전, 약한 아핀(±20°, 0.9–1.1배), 밝기·대비·감마·색조, 블러. 혈관을 인위적으로 휘게 하는 탄성 변형은 제외 |
| 최적화 | AdamW(lr 3e-4, wd 1e-4), OneCycle, AMP, batch 8, 80 에폭 |
| 추론 | 좌우·상하 반전 TTA, 검증셋 Dice가 최대인 임계값 사용 |

clDice는 혈관 **중심선의 연결성**을 직접 최적화하는 손실입니다. 가는 혈관이 끊기면 한 가닥이 여러 세그먼트로 쪼개져 굴곡도가 왜곡되기 때문에 추가했습니다.

### 5.3 세그먼트 추출
1. FOV 가장자리 10px 제외, 100px 미만 연결 요소 제거, 30px 미만 구멍 채움
2. 스켈레톤화 후 분기점 검출: 주변 8픽셀을 한 바퀴 돌 때 0→1 전이가 3회 이상인 픽셀
3. 분기점과 그 주변 1픽셀을 제거해 세그먼트로 분리 (분기점만 빼면 다른 가지끼리 대각선으로 붙어 합쳐지는 문제 방지)
4. 분기점에 붙은 12px 미만 잔가지 제거 (2회 반복)
5. 세그먼트 픽셀을 그래프 최장 경로로 정렬 → B-spline 평활 → 2px 간격 재샘플
6. 40px 미만 세그먼트 제외

### 5.4 굴곡도 지표

| 지표 | 정의 | 특징 |
|---|---|---|
| **DM** | 호 길이 / 현 길이 | 가장 직관적 (1 = 직선). C자와 S자를 구분하지 못함 |
| **ICM** | DM × (변곡점 수 + 1) | 여러 번 꺾이는 혈관을 더 높게 평가 |
| **SOAM** | Σ\|방향각 변화\| / 길이 (rad/px) | 자잘한 꼬임에 민감 |
| **CE** | ∫κ² ds / 길이 (1/px²) | 급하게 휘는 부분에 큰 가중치 |
| **TD** | (n−1)/n · (1/L) · Σ(하위 호/현 − 1) | Grisan(2008). 변곡점이 없는 C자형은 0 |

- 곡률 기반 지표(SOAM, CE, 변곡점)는 스플라인 경계 효과를 피하려고 세그먼트 양 끝 10px을 빼고 계산합니다.
- 변곡점: |κ| > 0.01인 같은 부호 곡률이 4샘플 이상 이어진 구간 사이의 부호 변화
- 영상 단위 값: 세그먼트 길이로 가중한 평균(`_lw`), 중앙값(`_med`), 상위 10%(`_p90`)

**계산 검증 (노트북 4절):** 모양을 아는 합성 곡선으로 확인했습니다.

| 곡선 | 기대값 | 결과 |
|---|---|---|
| 직선 (0°, 30°) | DM = 1, 변곡점 0 | DM 1.0000, 변곡점 0 |
| 원호 R = 100px | 곡률 0.01 | SOAM 0.0099, RMS 곡률 0.0101 |
| 사인파 3주기 (진폭 8·15·25px) | 변곡점 5개, 진폭에 따라 증가 | 변곡점 5개, DM 1.061 → 1.192 → 1.449 |

### 5.5 편향 점검과 질환군 비교
- **정답 vs 예측:** 테스트 200장에서 정답 마스크와 예측 마스크로 각각 굴곡도를 계산해 ICC(2,1, 절대 일치), Spearman ρ, Bland-Altman으로 비교합니다. (예측 − 정답) 차이가 질환군마다 다른지 Kruskal-Wallis로 검정하고 Holm 보정을 합니다.
- **질환군 비교:** 800장 전체의 정답 마스크로 계산한 굴곡도를 Kruskal-Wallis로 비교하고, 각 질환군을 정상과 Mann-Whitney로 비교합니다(지표 × 비교 전체에 Holm 보정).

## 6. 결과

### 6.1 학습
- 80 에폭, 에폭당 약 50초 (T4 ×2 기준 약 70분)
- 최고 검증 Dice 0.9306 (임계값 0.5, TTA 없음, 80 에폭)
- 선택된 임계값 **0.6** → 검증 Dice **0.9335** (TTA 적용)

### 6.2 분할 성능 (FIVES test 200장, FOV 내부)

| 질환군 | Dice | clDice | Sensitivity | Specificity | AUC | β0 오차 |
|---|---|---|---|---|---|---|
| 정상 | 0.918 | 0.932 | 0.898 | 0.993 | 0.995 | 29.9 |
| AMD | 0.933 | 0.936 | 0.925 | 0.994 | 0.996 | 16.4 |
| DR | 0.911 | 0.922 | 0.891 | 0.994 | 0.995 | 16.0 |
| 녹내장 | 0.866 | 0.876 | 0.849 | 0.995 | 0.990 | 13.2 |
| **전체** | **0.907** | **0.916** | **0.891** | **0.994** | **0.994** | **18.9** |

- 질환군 간 Dice 차이는 유의합니다 (Kruskal-Wallis H = 17.48, p = 0.0006). **녹내장군이 가장 낮고**, DR군은 전체 평균과 비슷합니다.
- β0 오차(예측과 정답의 연결 요소 수 차이)는 정상군에서 가장 큽니다. 가는 혈관이 많은 영상일수록 조각이 끊겨 연결 요소 수가 달라지는 것으로 보입니다.

### 6.3 정답 vs 예측 굴곡도 일치도 (test 200장)

영상당 세그먼트 수 중앙값: 정답 97개, 예측 84.5개

| 지표 | ICC(2,1) | Spearman ρ | 평균 차이 (예측 − 정답) | 질환군 간 차이 p (Holm) |
|---|---|---|---|---|
| DM_lw | 0.434 | 0.692 | −0.0009 | 0.079 |
| ICM_lw | 0.404 | 0.652 | −0.013 | 0.167 |
| **SOAM_lw** | **0.794** | **0.888** | −0.0006 | 0.015 |
| CE_lw | 0.630 | 0.817 | ≈ 0 | 0.019 |
| TD_lw | 0.662 | 0.788 | ≈ 0 | 0.167 |

ICC 해석 기준(Koo & Li 2016): 0.75 이상 좋음, 0.5–0.75 보통, 0.5 미만 낮음

- **SOAM만 "좋음"** 이고, CE·TD는 "보통", DM·ICM은 "낮음"입니다.
- DM의 ICC가 낮은 이유는 영상 간 DM 차이가 원래 작기 때문입니다(질환군 중앙값 1.025–1.030). 차이 폭이 좁으면 작은 분할 오차에도 영상 간 순위가 쉽게 바뀝니다.
- SOAM의 (예측 − 정답) 평균 차이는 정상 −0.0008, AMD −0.0004, DR −0.0004, 녹내장 −0.0009입니다. 예측이 전반적으로 약간 낮게 나오며, 질환군 간 차이가 통계적으로 유의하더라도(p = 0.015) **DR군에서 과대 측정되는 방향은 아닙니다.** 따라서 외부 데이터에서 DR 등급과 굴곡도의 양의 관계가 나오더라도, 그것을 DR 병변의 오분할 탓으로만 보기는 어렵습니다.

### 6.4 정답 마스크 기준 질환군 비교 (800장)

| 질환군 | DM_lw | ICM_lw | SOAM_lw | 혈관 밀도 | 세그먼트 수 |
|---|---|---|---|---|---|
| 정상 | 1.0297 | 1.453 | 0.0079 | 0.115 | 113 |
| AMD | 1.0251 | 1.375 | 0.0069 | 0.098 | 97 |
| DR | 1.0285 | 1.457 | 0.0075 | 0.095 | 80 |
| 녹내장 | 1.0264 | 1.411 | 0.0069 | 0.092 | 86 |

(중앙값. CE·TD는 소수점 넷째 자리 반올림 시 0.0001–0.0002로 표에서 생략)

정상군과 비교한 Holm 보정 p값:

| 지표 | AMD vs 정상 | DR vs 정상 | 녹내장 vs 정상 |
|---|---|---|---|
| DM_lw | 0.0006 | 1.000 | 0.002 |
| ICM_lw | 0.0006 | 1.000 | 0.380 |
| SOAM_lw | < 0.0001 | 0.021 | < 0.0001 |
| CE_lw | < 0.0001 | 0.012 | < 0.0001 |
| TD_lw | < 0.0001 | 1.000 | 0.0001 |
| 혈관 밀도 | < 0.0001 | < 0.0001 | < 0.0001 |

- 모든 지표에서 네 군 간 차이가 유의합니다 (Kruskal-Wallis p < 0.001).
- **DR군은 정상군보다 더 구불구불하지 않았습니다.** DM·ICM·TD는 차이가 없었고, SOAM·CE는 오히려 DR군이 약간 낮았습니다.
- 반면 **혈관 밀도와 세그먼트 수는 질환군 모두 정상군보다 뚜렷이 낮았습니다.** 가는 혈관이 덜 보이면 길이 가중 평균이 굵고 곧은 혈관 쪽으로 기울 수 있습니다. 따라서 질환군 간 굴곡도 차이의 일부는 "보이는 혈관의 구성"이 달라진 결과일 수 있고, 혈관 밀도를 보정한 분석이 필요합니다. 이 보정은 2차 노트북에서 합니다.

## 7. 실행 방법 (Kaggle)

1. **Settings → Accelerator:** `GPU T4 x2` 또는 `GPU P100`
2. **Settings → Internet:** On (smp 설치, ImageNet 가중치, kagglehub 다운로드에 필요)
3. **Add Input:** FIVES 데이터셋. 붙이지 않으면 `CFG.FIVES_KAGGLEHUB`로 자동 다운로드를 시도합니다.
4. **Save Version → Save & Run All**로 실행합니다. 학습에 약 70분이 걸리므로 백그라운드 실행을 권장합니다.
5. 2차 노트북에서 쓰려면 이 노트북의 출력(`unet_fives_best.pt`, `fives_gt_vs_pred_agreement.csv`)이 저장되어 있어야 합니다. 대화형 세션으로만 실행하면 세션이 끝날 때 출력이 사라집니다.

> 현재 Kaggle은 연결한 데이터를 `/kaggle/input/datasets/<소유자>/<이름>`, `/kaggle/input/competitions/<대회>` 같은 하위 폴더에 붙입니다. FIVES는 폴더 이름과 관계없이 파일명 패턴으로 찾습니다.

### 주요 설정 (`CFG`)

| 항목 | 값 | 설명 |
|---|---|---|
| `IMG_SIZE` | 1024 | FOV 지름 정규화 크기 |
| `IN_MODE` | `'rgb'` | `'rgb'`(LAB-L CLAHE) 또는 `'green'` |
| `EPOCHS` / `BATCH` / `CROP` | 80 / 8 / 512 | 학습 설정 |
| `CLDICE_W` | 0.5 | clDice 가중치 |
| `FOV_ERODE` | 10 | 굴곡도 계산 시 FOV 가장자리 제외 폭(px) |
| `MIN_OBJ` / `MIN_HOLE` | 100 / 30 | 작은 조각 제거, 작은 구멍 채움(px) |
| `SPUR_LEN` / `MIN_SEG_LEN` | 12 / 40 | 잔가지 제거, 최소 세그먼트 길이(px) |
| `STEP` / `SPLINE_S` | 2.0 / 0.5 | 재샘플 간격, 스플라인 평활 강도 |
| `KAPPA_MIN` / `MIN_RUN` / `END_TRIM` | 0.01 / 4 / 5 | 변곡점 판정, 양 끝 제외 샘플 수 |

후처리 파라미터의 픽셀 값은 모두 `IMG_SIZE = 1024` 기준입니다. 2차 노트북도 반드시 같은 값을 써야 결과를 비교할 수 있습니다.

## 8. 출력 파일 (`/kaggle/working`)

| 파일 | 내용 |
|---|---|
| `unet_fives_best.pt` | 학습된 U-Net 가중치 (2차 노트북 입력) |
| `train_history.csv` | 에폭별 손실, 검증 Dice |
| `fives_test_segmentation_metrics.csv` | 테스트 영상별 분할 지표 |
| `fives_gt_image_tortuosity.csv` | 정답 마스크 기준 영상별 굴곡도 (800장) |
| `fives_gt_segments.csv.gz` | 정답 마스크 기준 세그먼트별 굴곡도 |
| `fives_test_pred_image_tortuosity.csv` | 예측 마스크 기준 영상별 굴곡도 (test 200장) |
| `fives_gt_vs_pred_agreement.csv` | 정답–예측 일치도, 질환군별 편향 (2차 노트북 입력) |
| `figures/01_fives_examples.png` | 질환군별 원본과 정답 마스크 |
| `figures/02_training.png` | 학습 곡선 |
| `figures/03_fives_test_examples.png` | 질환군별 예측과 오류 지도 |
| `figures/05_bland_altman.png` | 정답 vs 예측 굴곡도 Bland-Altman |
| `figures/05_fives_group_comparison.png` | 정답 기준 질환군별 굴곡도 분포 |


## 9. 참고 문헌

- Ronneberger O, Fischer P, Brox T. U-Net: Convolutional networks for biomedical image segmentation. *MICCAI* 2015.
- Shit S, et al. clDice: A novel topology-preserving loss function for tubular structure segmentation. *CVPR* 2021.
- Jin K, et al. FIVES: A fundus image dataset for artificial intelligence based vessel segmentation. *Scientific Data* 2022.
- Grisan E, Foracchia M, Ruggeri A. A novel method for the automatic grading of retinal vessel tortuosity. *IEEE Transactions on Medical Imaging* 2008.
- Bullitt E, et al. Measuring tortuosity of the intracerebral vasculature from MRA images. *IEEE Transactions on Medical Imaging* 2003.
- Hart WE, et al. Measurement and classification of retinal vascular tortuosity. *International Journal of Medical Informatics* 1999.
- Koo TK, Li MY. A guideline of selecting and reporting intraclass correlation coefficients for reliability research. *Journal of Chiropractic Medicine* 2016.

## 10. 기술 스택

PyTorch, segmentation_models_pytorch, albumentations, OpenCV, scikit-image, SciPy, scikit-learn, pandas, joblib, matplotlib, kagglehub
