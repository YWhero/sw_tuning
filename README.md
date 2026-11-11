# GR00T N1.7 학습 · 시각화

최상위 `.sh`는 아래 여섯 개입니다. Conda `tuning` 환경과 CUDA/import 경로를 자동 설정합니다.

| 실행기 | 용도 | 기본값 |
|---|---|---|
| `groot-train.sh` | 원본 baseline: 비전·LLM 동결, 액션헤드 학습 | 100000스텝, 배치64 |
| `vision-kl.sh` | 비전타워 전체만 학습, LLM index15 attention KL | 1000스텝, 배치8 × 누적8 |
| `vision-hybrid.sh` | 비전타워 전체만 학습, feature hybrid + anchor + 상단 배경 attention 억제 | 1000스텝, 배치8 × 누적8 |
| `attention-map.sh` | 실제 LLM attention 히트맵 MP4 | 기본 LLM index15(16번째 층) |
| `cosine-map.sh` | 이미지·instruction feature cosine 히트맵 MP4 | 최종 VLM feature |
| `start_labeling.sh` | 통합 세그멘테이션 라벨링툴 실행 | 헤드·양쪽 손목캠, 포트8769 |

두 비전 실행기는 LLM·액션 모듈을 동결합니다. Hybrid는 마지막 두 블록만 학습하는 과거 실험과 다릅니다. 새 학습은 모두 **letterbox=false**입니다. 시각화는 지정한 체크포인트에 저장된 전처리를 유지합니다.

## 파일 상단에서 설정하기

다섯 실행기의 `여기서 설정하세요` 영역을 수정한 뒤 `./파일명.sh`로 실행합니다.
시각화는 `DATASET`, `EPISODES=(71 101)`, `CHECKPOINT`(또는 `VLM_WEIGHTS`), `OUT`을 설정하세요.
가중치 경로를 모두 비우면 원본 모델을 시각화합니다. 출력 폴더는 기존 결과와 겹치지 않게 지정합니다.
학습은 `DATASET`, `EXP_NAME`, 스텝·배치·누적·학습률·GPU를 설정합니다.
학습 환경변수와 기존 명령행 옵션도 지원하며, 명령행 옵션이 파일 설정보다 우선합니다.

## 학습

```bash
cd /home/ubuntu/mission_reasoning/tuning
./groot-train.sh --dataset Refilling_Dispenser_merged_932_934_lerobot_v21 --name baseline-new --check
# 검사 후 --check를 --background 또는 --foreground로 바꾸어 실행
./vision-kl.sh --dataset /path/to/verified-mask-dataset --name vision-kl-new --check
./vision-hybrid.sh --dataset /path/to/verified-mask-dataset --name vision-hybrid-new --check

# 메모리를 줄이면서 유효 배치64 유지
GLOBAL_BATCH_SIZE=8 GRADIENT_ACCUMULATION_STEPS=8 ./groot-train.sh \
  --dataset Refilling_Dispenser_merged_932_934_lerobot_v21 --name baseline-new --background
```

공통 recipe는 왼쪽 헤드캠 + 양팔(`arm_left`/`arm_right`, `sub_tasks`)입니다. 실행 중인 3카메라 실험은 해당 `logs/<실험>/` 전용 실행기와 설정을 유지합니다.

Task802 원본은 `data/802_mcap_lerobot_v21`입니다. 원본의 `left_arm`/`right_arm`, `task`, `object_1` 마스크는 검증된 전용 adapter/config가 필요하므로 공통 런처는 원본 경로의 직접 실행을 계속 차단합니다. `arr_0`는 물체+손 합집합이며 target-only 마스크로 사용하면 안 됩니다. 기존 802 실험 재현은 해당 로그 폴더의 전용 실행기를 사용합니다.

`MAX_STEPS`, `GLOBAL_BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`, `LEARNING_RATE`, `SAVE_STEPS`, `OUTPUT_DIR`, `GPU` 등 기존 환경변수를 지원합니다. 비전 기본 LR은 1e-5, 액션은 1e-4입니다. `--check`는 입력 검증만 수행합니다. 공통 런처는 다른 학습이 실행 중이면 신규 학습을 차단합니다.

## MP4 생성

```bash
./attention-map.sh \
  --checkpoint checkpoints/vision_runs/<실험>/checkpoint-1000 \
  --dataset /path/to/verified-dataset --episodes 71 101 \
  --out results/language_attention/<실험>_attention

./cosine-map.sh \
  --checkpoint checkpoints/vision_runs/<실험>/checkpoint-1000 \
  --dataset /path/to/verified-dataset --episodes 71 101 \
  --out results/language_attention/<실험>_cosine
```

`--out`은 새 폴더여야 합니다. 기본 모델은 `nvidia/GR00T-N1.7-3B`이며 `--base`로 바꿀 수 있습니다. 전체 체크포인트 대신 `--vlm-weights checkpoints/vision/<실험>-vlm.safetensors`도 지원합니다. 해당 모델/카메라에 맞는 데이터셋 입력을 지정합니다.

Attention은 마지막 유효 prompt query → image keys의 head별 image-conditional softmax 평균입니다. Cosine은 이미지 패치와 instruction feature의 방향 유사도로, `sigmoid(10*cosine)` 색상으로 표시합니다. 실제 attention 확률과는 다릅니다. 두 실행기 모두 기존 렌더러를 사용하고 MP4·NPZ·manifest를 저장합니다. 학습 실행기에서 영상을 자동 생성하지는 않습니다.

## 저장 · 보조 도구

```text
checkpoints/vision/<실험>-vlm.safetensors        VLM 전체 가중치
checkpoints/vision_runs/<실험>/checkpoint-<step>/   비전 재개용 전체 체크포인트
checkpoints/groot/<실험>/checkpoint-<step>/         액션 체크포인트
logs/<실험>.log                                  공통 학습 로그
scripts/activate-conda.sh                        수동 환경 활성화
scripts/action-train.sh                          튜닝된 비전으로 액션 학습
scripts/start_labeling.sh                        라벨링 서비스
scripts/legacy/                                 과거 Q/K·last2 실험 실행기
```

수동 환경 설정은 `source scripts/activate-conda.sh`입니다. 보조 액션 학습은 환경 활성화 후 `scripts/action-train.sh --dataset ... --name ... --vlm-weights ...`로 실행합니다. 과거 실험 스크립트는 기존 저수준 CLI를 유지합니다. 재개는 동일 설정에 `RESUME=1`을 지정하며 기존 실험/가중치를 덮어쓰지 않습니다.

공통 코드는 `tools/pipeline.py`, 시각화는 `tools/language_attention_videos.py`입니다. `Isaac-GR00T/`는 독립 upstream 체크아웃에 로컬 학습 기능 수정이 포함되어 있으므로 새 clone만으로 기능이 복구되지는 않습니다. 데이터·체크포인트·결과는 이번 정리에서 이동하거나 삭제하지 않았습니다. 이전 셸/문서는 `backups/shell_cleanup_*/`에 보존했습니다.

## 라벨링툴 기본 실행

```bash
cd /home/ubuntu/mission_reasoning/tuning
bash start_labeling.sh data/Refilling_Dispenser_lerobot_v21
```

VS Code에서 **8769** 포트를 전달하고 <http://localhost:8769/>를 엽니다. 이미 실행 중인 서버는 재사용합니다. 현재 기본 데이터는 `Refilling_Dispenser_lerobot_v21`입니다.

- 왼쪽 헤드캠·양쪽 손목캠을 한 페이지에서 공통 Include / Exclude / 되돌리기 / 저장 버튼으로 작업합니다.
- `저장 → 다음 에피소드` 또는 좌우 화살표로 이동합니다. 미저장 변경은 이동 전에 확인합니다.
- 카메라마다 한 프레임에 예시 포인트를 저장하고, `SAM 마스크 생성` 버튼으로 현재 에피소드의 저장된 예시를 전파합니다. 카메라별 진행 상태와 완료 여부를 표시하며, 완료 후 직접 마스크 검토를 진행합니다. 검토용 MP4는 생성하지 않습니다.
- 화면: `tools/refilling_labeling_page.html`; 단일 포트 서버: `tools/refilling_labeling_dashboard.py`; 카메라 설정: `configs/labeling_refilling_cam_*.json`.
- 로그: `logs/labeling_refilling_dashboard.log`, `logs/labeling_refilling_cam_*.log`.

앞으로 라벨링툴 실행은 이 명령과 통합 화면을 기본으로 사용합니다.

여러 라벨은 `새 라벨 이름` → `+ 라벨 추가`로 만들고 `작업 라벨`에서 선택합니다. 각 라벨에 Include/Exclude 포인트를 따로 지정한 뒤 저장·SAM 생성을 실행합니다. 카메라별 한 예시 프레임에 여러 라벨을 지정할 수 있으며, 대상이 없는 카메라에는 해당 라벨 포인트를 작성하지 않아도 됩니다. `선택 라벨만 표시`로 결과를 분리해 확인합니다. 라벨별 결과는 작업 폴더의 `object_masks.npz`(객체 ID 키)와 `annotations.json`(ID·이름 대응)에 보존됩니다. `masks.npz`의 `arr_0`는 모든 라벨의 합집합이므로 특정 라벨 학습에는 객체별 배열을 선택해야 합니다.

`선택 라벨 제거`는 현재 에피소드의 세 카메라에서 해당 라벨의 예시 포인트를 제거합니다. 확인 후 `저장`으로 반영하며, 저장 전 `되돌리기`로 복구할 수 있습니다. 마지막 라벨까지 제거할 수 있으며 빈 예시는 SAM 생성 대상에서 제외됩니다. 남은 라벨의 마스크는 SAM을 다시 생성하세요. 기존 작업 이력 파일은 보존됩니다.

다른 데이터셋은 `bash start_labeling.sh /절대/경로/데이터셋`으로 실행합니다. 상대 경로도 지원합니다. LeRobot v2.1 형식의 `meta/info.json`, `meta/episodes.jsonl`, 왼쪽 헤드캠 및 양쪽 손목캠 영상이 필요합니다. `meta/modality.json`의 video original_key를 사용하며 없으면 기본 카메라 키를 사용합니다. 같은 데이터셋은 재사용하고, 다른 데이터셋은 기존 SAM 생성이 완료된 경우에만 라벨링 서버를 전환합니다. 마스크와 예시는 지정한 데이터셋의 `labels/sam2_semantic`에 저장됩니다.

### SAM 모델 수명

`scripts/start_labeling.sh <데이터셋>` 실행 시 통합 서버가 SAM 모델 하나를 GPU에 미리 로드하며 준비 완료 후 접속 주소를 출력합니다. 세 카메라 작업은 이 모델을 공유하여 순차 처리합니다. 모델은 작업이 끝나도 유지되며, 각 작업의 영상 추출·추론 상태는 새로 생성됩니다. 로그는 `logs/labeling_sam.log`입니다.

종료 및 SAM GPU 메모리 해제: `bash start_labeling.sh --stop`. 실행 중인 마스크 작업도 중단되므로 저장된 예시로 다시 생성해야 합니다. 브라우저 탭을 닫는 것만으로는 서버를 종료하지 않습니다. 학습 세션은 유지됩니다.

### 기존 마스크를 보존하며 보완하기 (현재 동작)

- 변경 없는 라벨의 `object_masks.npz` 배열은 그대로 복사합니다. 새 라벨과 포인트가 바뀐 라벨만 SAM으로 전체 프레임에 전파합니다. `incremental.json`에 재사용·생성·삭제된 객체 ID가 기록됩니다.
- 프레임 슬라이더로 이동해도 기존 포인트를 유지합니다. 놓친 프레임에서 작업 라벨을 선택하고 Include/Exclude 포인트를 추가한 뒤 `SAM 마스크 생성`을 누르면 됩니다. 같은 라벨의 여러 프레임 포인트를 함께 사용합니다.
- `편집 이미지`에서 카메라를 선택하고 `이 이미지의 라벨 제거`를 누르면 그 카메라의 해당 라벨만 제거합니다. `현재 프레임 포인트 제거`는 다른 프레임의 보완 포인트를 유지합니다. 이미지 클릭 시 편집 이미지도 해당 카메라로 선택됩니다.
- 삭제/수정은 저장 전 되돌릴 수 있습니다. 저장 후 생성하면 변경된 결과가 별도 작업 폴더에 기록되며 이전 작업 이력은 보존됩니다. 다른 라벨은 유지되지만, 직접 보완한 라벨 자체는 전체 영상에서 다시 계산됩니다.
- 위 동작은 기존의 ‘카메라마다 한 프레임만 지정’ 방식과 ‘세 카메라 전체 라벨 삭제’ 기본 동작을 대체합니다.



### Hybrid 상단 배경 억제 기본값

`vision-hybrid.sh`는 `UPPER_BACKGROUND_ATTENTION_WEIGHT=0.1`을 기본으로 사용합니다. LLM index15의 실제 image-conditional attention에 상단1→하단0 선형 위치 가중치를 곱하고 target 마스크와 전혀 겹치지 않는 패치만 억제합니다. 파일 상단에서 계수를 조절하고, 0으로 두면 기존 hybrid loss만 사용합니다. tmux 실행에도 전달되며 실제 계수는 run manifest와 모델 config에 기록됩니다. KL/baseline 실행기의 손실은 변경하지 않습니다.

## 바로 실행할 수 있는 기본 경로

- Baseline: 병합 Refilling 데이터, 원본 GR00T.
- KL/Hybrid: `logs/refilling_visual_hybrid_upperbg_nopad_1000_20261111/inputs/head`의 물병+오른손 마스크 snapshot, 헤드캠 단독 recipe. 현재 사용 승인된 마스크를 사용하도록 `ALLOW_LEGACY_MASKS=1`; 정식 시각 검토 완료를 의미하지 않습니다.
- Attention/Cosine: 원본 Refilling 데이터, 에피소드0/1, 완료된 Refilling KL no-padding checkpoint1000. CHECKPOINT를 비우면 지정된 VLM_WEIGHTS 사용; 둘 다 비우면 원본 모델. CLI 가중치를 지정하면 파일의 가중치 기본값을 생략합니다.
- 라벨링: 인자 없이 실행하면 원본 Refilling 데이터.

실험명과 영상 출력 폴더는 실행 시각을 붙여 생성합니다. 재개할 때는 기존 EXP_NAME을 명시하세요. 새 기본값은 진행 중인 학습에 적용되지 않습니다.

## VLM 전체 저장 및 액션 학습

KL/hybrid 완료 시 `tools/extract_vlm.py`로 `backbone.*` 전체(비전 인코더·merger·LLM 및 백본에 포함된 출력 모듈)를 `checkpoints/vision/<실험>-vlm.safetensors`에 저장합니다. 액션헤드는 제외합니다. 학습 재개용 전체 체크포인트는 그대로 보존합니다. LLM이 동결된 실험도 LLM 가중치를 함께 저장합니다. 학습 대상 레이어 자체는 변경하지 않습니다.

```bash
./scripts/action-train.sh --check
./scripts/action-train.sh --vlm-weights checkpoints/vision/<실험>-vlm.safetensors --background
```

액션 학습은 원본 GR00T에 VLM 전체를 엄격하게 로드하고 backbone 전체를 freeze/eval하여 액션헤드만 학습합니다. 기본 데이터는 병합 Refilling, 기본 VLM은 Refilling hybrid-upperbg 1000스텝입니다. 공통 recipe는 헤드캠이며 3카메라 전용 실험과 구분됩니다. `groot-train.sh`는 계속 원본 baseline입니다.

새 VLM 파일에는 `--vlm-weights`를 사용합니다. 기존 비전 전용 파일과 `tools/extract_vision.py`는 과거 실험 재현용으로 보존합니다. 시각화는 기존 `--vision-weights`도 지원하지만 새 VLM 액션 학습에 비전 전용 파일을 주면 전체 백본 키 검증에서 거부합니다. 상위 공통 런처의 옛 `--vision-weights` 옵션 이름은 `--vlm-weights`의 호환 별칭입니다.

### 비전튜닝 시작 모델 고정

`vision-kl.sh`와 `vision-hybrid.sh`는 원본 사전학습 `nvidia/GR00T-N1.7-3B`에서 시작합니다. 셸의 BASE_MODEL_PATH는 이 값으로 고정하며, 공통 런처는 다른 `--base`와 기존 튜닝 VLM/비전 overlay를 거부합니다. 명시적 동일 실험 재개(`RESUME=1`)는 기존 동작을 유지합니다. 액션 학습의 VLM 이식 경로는 유지합니다.
