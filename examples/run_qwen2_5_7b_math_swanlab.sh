set -x

export SWANLAB_API_KEY=${SWANLAB_API_KEY}
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-/mnt/train/home/wuwx/swanlab_logs}
export SWANLAB_MODE=${SWANLAB_MODE:-cloud}

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0

SYSTEM_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{} if it is a math problem."""

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    data.system_prompt="${SYSTEM_PROMPT}" \
    data.train_files=${TRAIN_FILES:-hiyouga/math12k@train} \
    data.val_files=${VAL_FILES:-hiyouga/math12k@test} \
    data.max_prompt_length=${MPL:-2048} \
    data.max_response_length=${MRL:-2048} \
    worker.actor.model.model_path=${MODEL_DIR:-/mnt/models/models/public/Qwen/Qwen2.5/Qwen2.5-7B-Instruct/} \
    worker.rollout.enable_chunked_prefill=false \
    worker.rollout.max_num_batched_tokens=${MNBT:-8192} \
    trainer.total_episodes=${TE:-30} \
    trainer.logger=['console','swanlab'] \
    trainer.experiment_name=${EXPERIMENT:-qwen2_5_7b_math} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${NNODES:-1} \
    trainer.save_freq=${SAVE_FREQ:-50} \
    trainer.save_checkpoint_path=${OUTPUT_DIR:-/mnt/train/home/wuwx/easyr1_results} \
    trainer.load_checkpoint_path=${LOAD_CHECKPOINT_PATH:-null} \
