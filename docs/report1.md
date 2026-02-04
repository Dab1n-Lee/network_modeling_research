# Emergent Pitch Geometry in Self-Supervised Audio Representations  
## Internal Research Report (MVP1)

### Author
이다빈

### Date
2026-02-04

---

## 1. 연구 배경 및 목적

**자기지도학습(Self-Supervised Learning, SSL)** 기반 오디오 모델이   **명시적인 pitch supervision 없이도 음높이(pitch)에 민감한 표현 구조를 형성하는지**를 검증하는 것을 목적으로 함.

특히,
- 미세 주파수 차이(cent 단위)가
- 모델의 latent representation 공간에서
- **연속적·선형적·구조적인 축(pitch axis)**으로 나타나는지를 **표현기하(Representational Geometry)** 관점에서 정량 분석

향후 **인간 pitch hyperacuity** 및 **선천적/후천적 청각 처리 가설**과의 연결을 목표로 한 **모델 기반 가설 생성 단계(MVP1)**이다.

---

## 2. 사용 데이터 (Stimuli)

### 2.1 자극 구성

- 기준음: **A4 = 440 Hz**
- Pitch variation:  
  - ±1, ±2, ±3, ±5, ±10, ±20 cents  
  - + 0 cent (reference)
- 총 pitch 조건: **13개**
- 음색 조건:
  - **Pure tone**
  - **Harmonic complex tone**
- 총 자극 수: **26 stimuli (13 × 2)**

### 2.2 오디오 사양

- Sampling rate: **48 kHz**
- Duration: 고정 길이 (모델 입력 요구사항 충족)
- Loudness: RMS 정규화

### 2.3 메타데이터

- `stimuli_meta.csv`
  - stimulus_id
  - condition (pure / harmonic)
  - cents
  - frequency (Hz)

---

## 3. 모델 및 가중치

### 3.1 모델

- **AudioMAE (Masked Autoencoder for Audio)**
- ViT-Base backbone
- Encoder depth: **12 layers**
- Embedding dimension: **768**

### 3.2 학습 방식

- Self-supervised pretraining
- Reconstruction objective (masked spectrogram reconstruction)
- **Pitch 관련 supervision 없음**

### 3.3 가중치

- **ViT-B, AudioSet-2M pretrained checkpoint**
- 원전 GitHub repository 기반 로딩
- Fine-tuning 및 재학습 없음 (pure inference)
    - https://drive.google.com/file/d/1ni_DV4dRf7GxM8k-Eirx71WP9Gg89wwu/view?usp=share_link

---

## 4. 분석 방법

### 4.1 Embedding 추출

- 각 stimulus를 모델에 입력
- Encoder의 **모든 레이어(block_00 ~ block_11)** 에서 latent embedding 추출
- 결과 저장:
  - `audiomae_layerwise_embeddings.npz`

형태
- embeddings: (N=24, layers=12, dim=768)

---

### 4.2 Pitch Geometry 분석

각 레이어별로 다음 지표를 계산하였다.

#### (1) Pitch Sensitivity (Slope)
- 정의:
  - Representation distance ~ Δcents 선형 회귀 기울기
- 의미:
  - pitch 변화에 대한 민감도

#### (2) Pitch Correlation (Pearson r)
- 정의:
  - Δcents 와 representation distance 간 상관계수
- 의미:
  - pitch-aligned structure 존재 여부

#### (3) Curvature Proxy
- 정의:
  - 연속 pitch 조건 간 representation angle
- 해석:
  - 180°에 가까울수록 local manifold가 선형적
  - pitch axis의 “직선화(flattening)” 정도

---

## 5. 결과

### 5.1 Layerwise Pitch Sensitivity

- 초기 레이어(block_00)에서 가장 높은 slope
- 중간 레이어에서 감소
- 후반 레이어(block_06 ~ block_11)에서 재상승

→ pitch 변화가 네트워크 전반에 걸쳐 **구조적으로 유지**

---

### 5.2 Layerwise Pitch Correlation

- 모든 레이어에서 **양의 Pearson r (≈ 0.08–0.09)**
- 후반 레이어에서 다시 증가

→ pitch는 주 목적 변수는 아니지만, **일관된 정렬 축으로 존재**

---

### 5.3 Curvature 변화

- 평균 curvature angle:
  - block_00: ~165.2°
  - block_11: ~166.7°
- 깊은 레이어로 갈수록 **180°에 접근**

→ pitch manifold가 **점점 선형화**

---

## 6. 정량 요약 (5 lines)

1. AudioMAE는 pitch supervision 없이도 Δcents에 비례하는 latent distance 구조를 형성함.
2. Pitch sensitivity slope는 초기 및 후반 레이어에서 높아지는 U-shape 패턴을 보임.
3. 모든 레이어에서 Pearson r ≈ 0.08–0.09의 안정적인 양의 상관을 확인함.
4. Curvature 분석 결과, 깊은 레이어로 갈수록 pitch manifold가 선형화됨.
5. 이는 SSL 오디오 모델 내에 **연속 pitch axis가 자발적으로 형성됨**을 시사.

---

## 7. 생성된 결과물

### CSV
- `audiomae_layerwise_geometry.csv`

### Figures
- `layerwise_curvature_mean.png`
- `layerwise_pearson_r.png`
- `layerwise_slope.png`

(모두 outputs/analysis/audiomae/ 경로에 저장됨)

---

## 8. 요약 및 향후 방향

본 결과는 **SSL 오디오 모델이 pitch를 명시적으로 학습하지 않아도 연속적이고 구조적인 pitch representation을 형성한다는 증거**를 제공

향후 연구에서:
- Pure vs Harmonic 조건 분리 분석
- CLAP 등 semantic SSL 모델과의 geometry 비교
- 인간 JND / MEG / fMRI 데이터와의 정합 분석

으로 확장 가능성 존재.

---

본 Research 에서는 AudioMAE 를 사용을 하였으며, 향후 재현하기 위해서는 git clone --recurse-submodules <REPO_URL> 을 통해서 불러오는 것을 권장함.