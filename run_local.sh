#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET="${DATASET:-badminton}"
MODEL="${MODEL:-wasb}"
SPLIT="${SPLIT:-test}"
DEVICE="${DEVICE:-cpu}"
TORCH_VARIANT="${TORCH_VARIANT:-}"
DOWNLOAD_WEIGHTS="${DOWNLOAD_WEIGHTS:-0}"
VIDEO_BATCH_SIZE="${VIDEO_BATCH_SIZE:-4}"
TEST_NUM_WORKERS="${TEST_NUM_WORKERS:-0}"
INFERENCE_VIDEO_NUM_WORKERS="${INFERENCE_VIDEO_NUM_WORKERS:-0}"
VIS_RESULT="${VIS_RESULT:-0}"
SAVE_VIS_FRAMES="${SAVE_VIS_FRAMES:-0}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_ROOT/.cache/matplotlib}"

mkdir -p "$PROJECT_ROOT/datasets" "$PROJECT_ROOT/pretrained_weights" "$PROJECT_ROOT/outputs" "$MPLCONFIGDIR"

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

if [ -z "$TORCH_VARIANT" ]; then
  if [ "$DEVICE" = "cuda" ]; then
    TORCH_VARIANT="cu121"
  else
    TORCH_VARIANT="cpu"
  fi
fi

python -m pip install --upgrade pip setuptools wheel
pip install 'numpy<2' 'hydra-core==1.3.2' 'omegaconf==2.3.0' 'tqdm>=4.64' 'opencv-python-headless>=4.9,<5' 'pandas>=1.5,<2.3' 'matplotlib>=3.8,<4' pillow

if [ "$TORCH_VARIANT" = "cpu" ]; then
  pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2 torchvision==0.17.2
else
  pip install --index-url "https://download.pytorch.org/whl/$TORCH_VARIANT" torch==2.2.2 torchvision==0.17.2
fi

case "$MODEL" in
  wasb)
    WEIGHT_FILE="wasb_${DATASET}_best.pth.tar"
    EXTRA_ARGS=()
    ;;
  monotrack)
    WEIGHT_FILE="monotrack_${DATASET}_best.pth.tar"
    EXTRA_ARGS=("tracker=intra_frame_peak" "detector.postprocessor.use_hm_weight=False")
    ;;
  restracknetv2)
    WEIGHT_FILE="restracknetv2_${DATASET}_best.pth.tar"
    EXTRA_ARGS=("tracker=intra_frame_peak" "detector.postprocessor.use_hm_weight=False")
    ;;
  tracknetv2)
    WEIGHT_FILE="tracknetv2_${DATASET}_best.pth.tar"
    EXTRA_ARGS=("tracker=intra_frame_peak" "detector.postprocessor.use_hm_weight=False")
    ;;
  ballseg)
    WEIGHT_FILE="ballseg_${DATASET}_best.pth.tar"
    EXTRA_ARGS=("tracker=intra_frame_peak" "detector.postprocessor.use_hm_weight=False" "detector.step=1")
    ;;
  deepball)
    WEIGHT_FILE="deepball_${DATASET}_best.pth.tar"
    EXTRA_ARGS=("detector=deepball" "tracker=intra_frame_peak" "detector.step=1")
    ;;
  deepball_large)
    WEIGHT_FILE="deepball-large_${DATASET}_best.pth.tar"
    EXTRA_ARGS=("detector=deepball" "tracker=intra_frame_peak" "detector.step=1")
    ;;
  *)
    echo "不支持的 MODEL: $MODEL"
    exit 1
    ;;
esac

WEIGHT_PATH="$PROJECT_ROOT/pretrained_weights/$WEIGHT_FILE"
if [ ! -f "$WEIGHT_PATH" ]; then
  if [ "$DOWNLOAD_WEIGHTS" = "1" ]; then
    if ! command -v wget >/dev/null 2>&1; then
      echo "DOWNLOAD_WEIGHTS=1 需要系统安装 wget"
      exit 1
    fi
    (
      cd "$PROJECT_ROOT/src"
      bash setup_scripts/setup_weights.sh
    )
  else
    echo "缺少权重文件: $WEIGHT_PATH"
    echo "先执行: cd \"$PROJECT_ROOT/src\" && bash setup_scripts/setup_weights.sh"
    exit 1
  fi
fi

if [ ! -d "$PROJECT_ROOT/datasets/$DATASET" ]; then
  echo "缺少数据集目录: $PROJECT_ROOT/datasets/$DATASET"
  exit 1
fi

if [ "$DEVICE" = "cpu" ]; then
  DEVICE_ARGS=("runner.device=cpu" "runner.gpus=[]")
elif [ "$DEVICE" = "cuda" ]; then
  DEVICE_ARGS=("runner.device=cuda" "runner.gpus=[0]")
else
  echo "不支持的 DEVICE: $DEVICE"
  exit 1
fi

cd "$PROJECT_ROOT/src"
PYTHONPATH="$PROJECT_ROOT/src" python main.py \
  --config-name=eval \
  "dataset=$DATASET" \
  "model=$MODEL" \
  "runner.split=$SPLIT" \
  "detector.model_path=$WEIGHT_PATH" \
  "runner.vis_result=$VIS_RESULT" \
  "runner.save_vis_frames=$SAVE_VIS_FRAMES" \
  "dataloader.test_num_workers=$TEST_NUM_WORKERS" \
  "dataloader.inference_video_num_workers=$INFERENCE_VIDEO_NUM_WORKERS" \
  "dataloader.sampler.inference_video_batch_size=$VIDEO_BATCH_SIZE" \
  "${DEVICE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
