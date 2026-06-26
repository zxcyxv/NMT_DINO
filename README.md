# NMT_DINO: Old Assyrian Neural Machine Translation

NMT_DINO는 Kaggle **Deep Past Challenge**를 위해 작성한 Old Assyrian Akkadian transliteration → English translation 파이프라인입니다. 저자원 고대어 번역 문제에서 ByT5 기반 seq2seq 모델을 더 잘 활용하기 위해, unlabeled transliteration 데이터에 대한 자기 지도 학습과 MBR decoding, domain-specific 전처리/후처리를 실험합니다.

핵심 목표는 다음입니다.

> 약 1,500개의 정렬된 학습 번역과 수천 개의 unlabeled Old Assyrian transliteration을 활용해, 문헌 특유의 표기법과 결손 표기를 견디는 번역 시스템을 만드는 것.

## 프로젝트 개요

Deep Past Challenge는 고대 아시리아 상인들의 점토판 기록을 영어로 번역하는 저자원 기계번역 문제입니다. 입력은 Akkadian transliteration이고, 출력은 한 문장 단위의 영어 번역입니다. 평가는 corpus-level BLEU와 chrF++의 geometric mean으로 이루어집니다.

이 저장소는 다음 아이디어를 구현합니다.

- ByT5 계열 byte-level seq2seq 모델을 기본 번역기로 사용
- OARE/OA transliteration의 특수 표기, determinative, 결손 표기, 분수/수량 표기를 정규화
- Unlabeled `published_texts.csv`에 대해 DINO/denoising 계열 자기 지도 학습 실험 수행
- 초기 실험에서는 token-wise DINO와 decoder KD를 적용
- 개선 버전에서는 continuous latent view, encoder-only DINO, large projection head를 사용
- 제출 단계에서는 beam/sample candidate pool을 만들고 chrF++ 기반 MBR로 후보를 선택
- Kaggle Notebook 실행을 위해 single-GPU와 multi-GPU notebook 생성 스크립트 제공

## 저장소 구조

```text
nmt_dino/
  preprocessing.py                    Old Assyrian transliteration 전처리 공통 모듈
  postprocessing.py                   번역 출력 후처리 공통 모듈
  dino.py                             DINO projection head와 corruption utility
  encoder_ssl.py                      개선된 encoder-only latent-view SSL 공통 모듈

train_encoder_ssl.py                  개선 버전: encoder-only continuous-view DINO SSL
phase1_dino_ema_train.py              Multi-GPU DINO EMA + sequence KD 학습
phase1_dino_ema_train_single_gpu.py   Single-GPU DINO EMA + sequence KD 학습
phase1_dino_train2.py                 초기 DINO + reconstruction CE 실험

evaluate.py                           Original vs DINO model BLEU/chrF++ 평가
diagnose.py                           데이터, backbone, DINO head, 번역 품질 진단
submission_mbr.py                     DINO model + 외부 model 후보 pool + MBR 제출 생성
kaggle_cell_single_model_mbr.py       단일 모델 MBR Kaggle cell 버전

create_notebook.py                    Multi-GPU Kaggle notebook 생성
create_notebook_single.py             Single-GPU Kaggle notebook 생성
Kaggle_DINO_EMA_Pipeline.ipynb        생성된 multi-GPU Kaggle notebook
Kaggle_DINO_EMA_SingleGPU.ipynb       생성된 single-GPU Kaggle notebook
lb-35-9-ensembling-post-processing-baseline.ipynb
                                     기존 ensembling/post-processing baseline notebook

description.txt                       Kaggle competition brief 및 데이터 설명
requirements.txt                      로컬 실행용 Python 의존성
```

`evaluate.py`, `diagnose.py`, `submission_mbr.py`는 `nmt_dino/`의 공통 전처리/후처리 모듈을 사용합니다. 반면 Kaggle notebook 생성 스크립트는 competition 제출 편의를 위해 필요한 코드를 notebook 안에 복사하는 self-contained 구조를 유지합니다.

## 접근 방식

### 1. Transliteration 전처리

Old Assyrian transliteration에는 일반 NLP 데이터와 다른 표기 문제가 많습니다. 이 프로젝트의 전처리는 다음을 다룹니다.

- `sz`, `s,`, `t,` 등을 `š`, `ṣ`, `ṭ`로 변환
- `a2`, `e3`, subscript 숫자 등을 diacritic/ASCII 형태로 정규화
- 결손 표기, lacuna, `x`, `...`, bracket notation을 `<gap>`으로 통합
- determinative 표기 `{d}`, `{ki}`, `{m}` 등 보존 또는 표준화
- 분수/소수 수량 표기를 canonical form으로 변환
- OCR/출판물에서 온 불필요한 scribal mark 제거

공통 구현은 [nmt_dino/preprocessing.py](nmt_dino/preprocessing.py)에 있습니다.

### 2. 초기 DINO EMA Pretraining

주요 학습 스크립트는 [phase1_dino_ema_train.py](phase1_dino_ema_train.py)와 [phase1_dino_ema_train_single_gpu.py](phase1_dino_ema_train_single_gpu.py)입니다.

학습 구조:

```text
clean input      -> EMA teacher -> encoder projection p_T
corrupted input  -> student     -> encoder projection p_S

clean input      -> EMA teacher decoder -> p_T_dec
corrupted input  -> student decoder     -> p_S_dec
```

손실:

```text
L_total = lambda_dino * L_DINO + lambda_kd * L_KD
```

- `L_DINO`: teacher projection distribution과 student projection distribution 사이의 token-wise DINO loss
- `L_KD`: teacher decoder logits와 student decoder logits 사이의 sequence-level KL distillation
- teacher는 gradient로 학습하지 않고 student의 EMA로만 갱신
- corruption은 sequence length를 유지하는 byte replacement 방식으로 수행

이 구조는 unlabeled transliteration에서도 encoder/decoder가 Old Assyrian 표기 분포에 적응하도록 설계되었습니다.

다만 이 초기 구조에는 중요한 한계가 있습니다.

- 바이트 ID를 무작위 교체하면 언어적으로 가능한 표기가 아니라 깨진 문자열 view가 생성됩니다.
- DINO loss를 token-wise로 걸면 문장 의미보다 위치별 byte perturbation에 과하게 민감해질 수 있습니다.
- Decoder KD까지 동시에 넣으면 번역용 decoder가 self-supervised objective에 끌려가 supervised NMT 성능을 훼손할 수 있습니다.
- `proj_output=256`은 DINO의 pseudo-class 공간으로는 작아 표현 군집을 지나치게 뭉갤 수 있습니다.

### 3. 개선 버전: Encoder-only Latent-view SSL

[train_encoder_ssl.py](train_encoder_ssl.py)는 위 한계를 반영한 개선 버전입니다. Kaggle notebook 제출용이 아니라, 로컬/서버에서 별도로 실행하는 연구용 학습 엔트리포인트입니다.

핵심 변경:

- Discrete byte replacement를 제거하고 embedding 이후의 continuous latent space에 Gaussian noise와 dropout을 적용
- Decoder reconstruction/KD objective를 제거하고 encoder representation만 자기 지도 학습
- Token-wise loss 대신 attention mask 기반 mean pooling 문장 벡터에 DINO loss 적용
- Projection head 출력 차원을 기본 `65536`으로 확장
- Decoder와 shared embedding은 기본적으로 freeze하여 번역 능력 손상을 줄임
- 산출 checkpoint는 이후 labeled Akkadian-English pair로 supervised fine-tuning하는 것을 전제로 함

실행 예시:

```bash
python train_encoder_ssl.py \
  --model_path /path/to/base-byt5 \
  --data_path /path/to/published_texts.csv \
  --output_dir outputs/encoder_ssl \
  --d_model 1536 \
  --batch_size 8 \
  --grad_accum 4
```

학습 결과는 다음처럼 저장됩니다.

```text
outputs/encoder_ssl/final/
  student/                  encoder SSL이 적용된 seq2seq checkpoint
  teacher/                  EMA teacher checkpoint
  encoder_ssl_state.pt      projection head, center, config
  encoder_ssl_config.json
```

이 checkpoint는 곧바로 제출 모델이라기보다, 다음 supervised fine-tuning 단계의 초기값입니다.

### 4. Evaluation

[evaluate.py](evaluate.py)는 labeled `train.csv` subset에서 original model과 DINO-pretrained model을 비교합니다.

측정 항목:

- BLEU
- chrF++
- per-sample chrF++ 개선/악화 분석
- sample translation 출력 비교

실행 예시:

```bash
python evaluate.py \
  --original_path /path/to/base-byt5 \
  --dino_path /path/to/dino_ema_output/final \
  --data_path /path/to/train.csv
```

### 5. MBR Submission

[submission_mbr.py](submission_mbr.py)는 Kaggle 제출 파일을 생성합니다.

주요 단계:

1. test transliteration 전처리
2. DINO student model로 beam/sample 후보 생성
3. 외부 baseline model 후보와 pool 결합
4. chrF++ 기반 pairwise consensus와 source fidelity를 반영한 MBR 후보 선택
5. domain-specific post-processing 적용
6. `submission.csv` 저장

출력 후처리는 [nmt_dino/postprocessing.py](nmt_dino/postprocessing.py)에 모듈화되어 있습니다.

실행 예시:

```bash
python submission_mbr.py \
  --dino_model_path /kaggle/working/dino_ema_output/final/student \
  --output_dir /kaggle/working \
  --batch_size 2 \
  --num_beams 8 \
  --max_new_tokens 384
```

## Kaggle Notebook 생성

Kaggle Code Competition 제출을 위해 notebook 파일을 자동 생성합니다.

Multi-GPU notebook:

```bash
python create_notebook.py
```

Single-GPU notebook:

```bash
python create_notebook_single.py
```

생성된 notebook은 다음 흐름을 포함합니다.

1. Kaggle runtime 의존성 설치
2. training script를 notebook 내부 파일로 작성
3. Kaggle input 경로와 baseline model 자동 탐색
4. DINO EMA pretraining 실행
5. labeled data subset 평가
6. MBR submission 생성

Notebook은 Kaggle 제출 환경에서 단일 파일로 실행되도록 설계되어 있어, 일부 공통 로직을 내부 cell에 포함합니다.

## 설치

로컬에서 코드를 읽거나 일부 스크립트를 실행하려면:

```bash
pip install -r requirements.txt
```

Kaggle에서는 notebook이 필요한 패키지를 셀 내부에서 설치하도록 구성되어 있습니다.

주의:

- 실제 학습은 GPU 환경을 전제로 합니다.
- 기본 경로는 Kaggle input/working 디렉터리에 맞춰져 있습니다.
- 대용량 competition dataset과 model checkpoint는 이 저장소에 포함하지 않습니다.

## 데이터

데이터는 Kaggle Deep Past Challenge의 competition dataset을 사용합니다.

주요 파일:

- `train.csv`: document-level Akkadian transliteration과 English translation
- `test.csv`: sentence-level Akkadian transliteration
- `published_texts.csv`: translation이 없는 약 8,000개 published text metadata/transliteration
- `OA_Lexicon_eBL.csv`: Old Assyrian lexicon
- `publications.csv`: OCR/LLM postprocessed publication text

이 저장소의 [description.txt](description.txt)는 competition brief와 데이터 필드 설명을 보존한 문서입니다.

## 연구/개발 포인트

이 프로젝트에서 중점적으로 다룬 문제는 다음입니다.

- 저자원 번역에서 labeled data 부족을 unlabeled transliteration pretraining으로 보완
- 고대어 transliteration의 표기 불일치와 결손 표기를 모델 입력에 맞게 안정화
- ByT5의 byte-level 특성을 활용하면서도 domain-specific normalization을 유지
- Teacher-student EMA 구조로 noisy corruption에 대한 표현 일관성 학습
- 개선 버전에서는 noisy token이 아니라 continuous latent view 사이의 표현 일관성 학습
- BLEU/chrF++ 평가뿐 아니라 MBR decoding으로 제출 품질 개선
- Kaggle Notebook 환경의 제약 안에서 학습, 평가, 제출까지 재현 가능한 pipeline 구성
- 반복되던 전처리/후처리/DINO utility를 `nmt_dino/` 패키지로 분리해 CLI 스크립트의 중복을 줄임

## 한계와 다음 단계

- Kaggle notebook 생성용 스크립트는 self-contained 실행을 위해 일부 로직을 계속 포함합니다.
- 학습/평가 경로가 Kaggle 환경에 강하게 맞춰져 있어, 로컬 재현용 config layer가 있으면 좋습니다.
- 초기 DINO/sequence-KD 구조와 개선된 encoder-only SSL 구조의 ablation 결과가 별도 리포트로 정리되어 있지 않습니다.
- `published_texts.csv`와 `publications.csv`의 translation alignment를 더 체계화하면 학습 데이터 확장이 가능합니다.

## 요약

NMT_DINO는 단순히 baseline model을 fine-tuning하는 접근이 아니라, unlabeled 고대어 transliteration을 활용해 ByT5 representation을 domain-adapt하고, MBR decoding과 domain-specific post-processing으로 Kaggle 제출 품질을 높이려는 실험적 NMT 파이프라인입니다.

최종 결론은 초기 DINO 이식 방식에 구조적인 문제가 있었다는 것입니다. DINO의 핵심은 입력의 여러 view가 같은 대상을 가리킨다고 보고, encoder가 안정적인 표현을 만들도록 학습시키는 데 있습니다. 그런데 byte-level text에서 입력 byte를 무작위로 바꾸면 같은 문장의 다른 view라기보다 깨진 문자열이 됩니다. 이 경우 모델은 아카드어의 형태나 문맥을 배우기보다, 인위적인 노이즈를 견디는 쪽으로 학습될 수 있습니다.

더 큰 문제는 decoder까지 같은 자기 지도 학습 단계에 묶은 점입니다. DINO는 표현학습을 위한 목적 함수이고, 번역 decoder는 target 문장을 생성하기 위한 목적 함수에 맞춰져야 합니다. 둘을 unlabeled source text 위에서 동시에 움직이면 encoder 표현을 정리하려는 gradient와 decoder 생성을 보존하려는 gradient가 서로 충돌할 수 있습니다. 그 결과 학습이 표현 공간을 잘 정리하기보다, 번역 성능을 깎거나 국소 최적점/안장점 근처에서 불안정하게 머물 가능성이 있습니다.

따라서 이 프로젝트의 개선 방향은 DINO를 encoder 표현학습으로 제한하는 것입니다. [train_encoder_ssl.py](train_encoder_ssl.py)는 byte 교체 대신 embedding 이후의 연속 공간에 작은 노이즈와 dropout을 주고, decoder를 기본적으로 고정한 채 sentence-level encoder representation만 학습합니다. 이후 번역 성능은 labeled parallel corpus로 별도 fine-tuning하면서 맞추는 것이 더 일관된 설계입니다.
