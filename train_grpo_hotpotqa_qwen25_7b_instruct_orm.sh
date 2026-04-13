#!/bin/bash
set -euo pipefail

# =========================
# Fixed config (edit here)
# =========================
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONUNBUFFERED=1

# Platform env (multi-node only)
: "${PET_MASTER_ADDR:?PET_MASTER_ADDR is required}"
: "${PET_MASTER_PORT:?PET_MASTER_PORT is required}"
: "${PET_NODE_RANK:?PET_NODE_RANK is required}"
: "${PET_NNODES:?PET_NNODES is required}"

export MY_IP=$(hostname -I | awk '{ print $1 }')
export MASTER_ADDR=$PET_MASTER_ADDR
export MASTER_PORT=$PET_MASTER_PORT
export NODE_RANK=$PET_NODE_RANK
export WORLD_SIZE=$PET_NNODES
export GPUS_PER_NODE=8

# Paths
CODE_PATH=/nfs-153/chenjiaqi/Verl/verl2/searchr1
BASE_DIR=/nfs-153/chenjiaqi/Verl/verl2/grpo_base
JOB_NAME=grpo_base

train_files=$CODE_PATH/data/hotpotqa_search2/train/train.parquet
val_files=$CODE_PATH/data/hotpotqa_search2/test/test.parquet

MODEL_LOAD=/itdd-pfs/RMD/models/Qwen/Qwen3-8B
CHECKPOINT_SAVE=$BASE_DIR

LOG_DIR=$BASE_DIR/logs
LOG_FILE=$LOG_DIR/console/node_${NODE_RANK}_stdout.log
export WANDB_DIR=$BASE_DIR/wandb

# W&B
export WANDB_MODE=offline
export WANDB_API_KEY=replace_me
export WANDB_ENTITY=Itdd_risk_decision
WAND_PROJECT=RL-verl-search
EXPERIMENT_NAME=grpo_base

# Retriever config
# mode:
# - platform: use platform service url (recommended)
# - local: launch local retriever from index/corpus on every node
RETRIEVER_MODE=platform
PLATFORM_RETRIEVER_URL=
RETRIEVER_TOPK=3
RETRIEVER_PROBE_TIMEOUT=600
RETRIEVER_PROBE_INTERVAL=5

# Local retriever paths (only used when RETRIEVER_MODE=local)
WIKI_DIR=/nfs-151/disk3/wiki
INDEX_PATH=$WIKI_DIR/e5_Flat.index
CORPUS_PATH=$WIKI_DIR/wiki-18.jsonl
RETRIEVER_NAME=e5
RETRIEVER_MODEL=$WIKI_DIR/e5-base-v2
RETRIEVER_URL=$PLATFORM_RETRIEVER_URL

# Optional runtime injection from platform env
# If your platform injects SEARCH_RETRIEVER_URL, it will override script value.
if [[ -n "${SEARCH_RETRIEVER_URL:-}" ]]; then
  PLATFORM_RETRIEVER_URL="$SEARCH_RETRIEVER_URL"
fi
RETRIEVER_URL="$PLATFORM_RETRIEVER_URL"

# Cluster runtime
EXPECTED_NODES=$WORLD_SIZE
TIMEOUT=1800
RAY_DASHBOARD_ADDRESS=http://$MASTER_ADDR:8265

# =========================
# Prepare dirs / env
# =========================
mkdir -p "$LOG_DIR/console" 2>/dev/null
mkdir -p "$LOG_DIR/ray/node_${NODE_RANK}" 2>/dev/null
mkdir -p "$WANDB_DIR/wandb" 2>/dev/null
mkdir -p "$CHECKPOINT_SAVE" 2>/dev/null

exec > >(stdbuf -oL tee -a "$LOG_FILE") 2>&1

export PYTHONPATH=$CODE_PATH:${PYTHONPATH:-}
cd "$CODE_PATH"

echo "===== Platform Env Check ====="
echo "PET_MASTER_ADDR=$PET_MASTER_ADDR"
echo "PET_MASTER_PORT=$PET_MASTER_PORT"
echo "PET_NODE_RANK=$PET_NODE_RANK"
echo "PET_NNODES=$PET_NNODES"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "NODE_RANK=$NODE_RANK"
echo "WORLD_SIZE=$WORLD_SIZE"
echo "MY_IP=$MY_IP"
echo "RAY_DASHBOARD_ADDRESS=$RAY_DASHBOARD_ADDRESS"
echo "CODE_PATH=$CODE_PATH"
echo "BASE_DIR=$BASE_DIR"
echo "LOG_FILE=$LOG_FILE"
echo "WANDB_DIR=$WANDB_DIR"
echo "MODEL_LOAD=$MODEL_LOAD"
echo "CHECKPOINT_SAVE=$CHECKPOINT_SAVE"
echo "train_files=$train_files"
echo "val_files=$val_files"
echo "INDEX_PATH=$INDEX_PATH"
echo "CORPUS_PATH=$CORPUS_PATH"
echo "RETRIEVER_MODEL=$RETRIEVER_MODEL"
echo "RETRIEVER_URL=$RETRIEVER_URL"
echo "RETRIEVER_MODE=$RETRIEVER_MODE"

if [[ ! -f "$train_files" ]]; then
  echo "[ERROR] train file not found: $train_files"
  exit 1
fi
if [[ ! -f "$val_files" ]]; then
  echo "[ERROR] val file not found: $val_files"
  exit 1
fi
if [[ ! -d "$MODEL_LOAD" ]]; then
  echo "[ERROR] model path not found (directory expected): $MODEL_LOAD"
  exit 1
fi
if [[ "$RETRIEVER_MODE" == "platform" ]]; then
  # security checks for platform url
  if [[ -z "$PLATFORM_RETRIEVER_URL" ]]; then
    echo "[ERROR] PLATFORM_RETRIEVER_URL is empty."
    exit 1
  fi
  if [[ ! "$PLATFORM_RETRIEVER_URL" =~ ^https?:// ]]; then
    echo "[ERROR] PLATFORM_RETRIEVER_URL must start with http:// or https://"
    exit 1
  fi
  if [[ "$PLATFORM_RETRIEVER_URL" =~ @ ]]; then
    echo "[ERROR] PLATFORM_RETRIEVER_URL must not contain embedded credentials."
    exit 1
  fi
  if [[ "$PLATFORM_RETRIEVER_URL" =~ localhost|127\.0\.0\.1|::1 ]]; then
    echo "[ERROR] platform mode forbids localhost/127.0.0.1/::1 retriever url."
    exit 1
  fi
  RETRIEVER_URL="$PLATFORM_RETRIEVER_URL"

  # connectivity probe with retry
  if command -v curl >/dev/null 2>&1; then
    elapsed=0
    while true; do
      if curl --max-time 5 -sS -X POST "$RETRIEVER_URL" \
        -H "content-type: application/json" \
        -d '{"queries":["health check"],"topk":1}' >/dev/null; then
        echo "platform retriever is ready: $RETRIEVER_URL"
        break
      fi

      sleep "$RETRIEVER_PROBE_INTERVAL"
      elapsed=$((elapsed + RETRIEVER_PROBE_INTERVAL))
      echo "waiting retriever... elapsed=${elapsed}s url=$RETRIEVER_URL"
      if [[ "$elapsed" -ge "$RETRIEVER_PROBE_TIMEOUT" ]]; then
        echo "[ERROR] cannot reach platform retriever within ${RETRIEVER_PROBE_TIMEOUT}s: $RETRIEVER_URL"
        exit 1
      fi
    done
  else
    echo "[WARN] curl not found, skip retriever connectivity probe."
  fi
elif [[ "$RETRIEVER_MODE" == "local" ]]; then
  if [[ ! -f "$INDEX_PATH" ]]; then
    echo "[ERROR] index file not found: $INDEX_PATH"
    exit 1
  fi
  if [[ ! -f "$CORPUS_PATH" ]]; then
    echo "[ERROR] corpus file not found: $CORPUS_PATH"
    exit 1
  fi
  RETRIEVER_URL=http://127.0.0.1:8000/retrieve
  python3 search_r1/search/retrieval_server.py \
    --index_path "$INDEX_PATH" \
    --corpus_path "$CORPUS_PATH" \
    --topk "$RETRIEVER_TOPK" \
    --retriever_name "$RETRIEVER_NAME" \
    --retriever_model "$RETRIEVER_MODEL" \
    --faiss_gpu &
  RETRIEVER_PID=$!
  trap 'kill $RETRIEVER_PID >/dev/null 2>&1 || true' EXIT
  sleep 10
else
  echo "[ERROR] RETRIEVER_MODE must be 'platform' or 'local', got: $RETRIEVER_MODE"
  exit 1
fi

# =========================
# Multi-node only
# =========================
if [[ "$NODE_RANK" == "0" ]]; then
  echo "当前节点是 Master ($MY_IP)"

  ulimit -n 65535 || true
  ray stop --force >/dev/null 2>&1 || true

  ray start --head \
    --port="$MASTER_PORT" \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 --block &

  sleep 10

  echo "等待Ray集群响应..."
  if ! timeout "$TIMEOUT" bash -c "until ray status --address=$MASTER_ADDR:$MASTER_PORT >/dev/null 2>&1; do sleep 5; done"; then
    echo "[ERROR] Ray集群未在 ${TIMEOUT}s 内响应"
    ray stop --force || true
    exit 1
  fi

  echo "等待节点连接（预期: $EXPECTED_NODES）..."
  if ! timeout "$TIMEOUT" bash -c "\
    while :; do \
      alive_nodes=\$(ray status 2>/dev/null | awk '/Active:/{f=1;c=0} /Pending:/{f=0} f&&/node_/{c++} END{print c+0}'); \
      echo \"当前节点数: \$alive_nodes/$EXPECTED_NODES\"; \
      [ \$alive_nodes -ge $EXPECTED_NODES ] && break; \
      sleep 5; \
    done"; then
    echo "[ERROR] 未在 ${TIMEOUT}s 内达到预期节点数"
    ray status || true
    ray stop --force || true
    exit 1
  fi

  sleep 20

  ray job submit --address="$RAY_DASHBOARD_ADDRESS" \
    --runtime-env=verl/trainer/runtime_env.yaml \
    -- \
    python3 -m verl.trainer.main_ppo \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=512 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path="$MODEL_LOAD" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=5 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes="$WORLD_SIZE" \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.project_name="$WAND_PROJECT" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.total_epochs=15 \
    trainer.total_training_steps=1005 \
    trainer.default_local_dir="$CHECKPOINT_SAVE" \
    max_turns=4 \
    retriever.url="$RETRIEVER_URL" \
    retriever.topk="$RETRIEVER_TOPK" \
    2>&1 | tee "$EXPERIMENT_NAME.log"

  ray stop --force || true
else
  echo "当前节点是 Worker ($MY_IP)，连接到: $MASTER_ADDR"
  sleep 20
  ray stop --force >/dev/null 2>&1 || true

  RESOLVED_IP=$(getent hosts "$MASTER_ADDR" 2>/dev/null | awk '{print $1}' | head -1)
  if [[ -z "$RESOLVED_IP" ]]; then
    RESOLVED_IP=$(ping -c 1 "$MASTER_ADDR" 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | head -1)
  fi

  if [[ -n "$RESOLVED_IP" ]]; then
    FINAL_MASTER=$RESOLVED_IP
    echo "解析成功，使用 IP 连接: $FINAL_MASTER"
  else
    FINAL_MASTER=$MASTER_ADDR
    echo "解析失败，回退使用原始地址: $FINAL_MASTER"
  fi

  elapsed=0
  retry_interval=10
  while true; do
    if ray start --address="${FINAL_MASTER}:${MASTER_PORT}" --block; then
      break
    fi
    sleep "$retry_interval"
    elapsed=$((elapsed + retry_interval))
    if [[ "$elapsed" -ge "$TIMEOUT" ]]; then
      echo "[ERROR] Worker连接超时: ${TIMEOUT}s"
      exit 1
    fi
    echo "Worker重试连接... elapsed=${elapsed}s"
  done
fi
