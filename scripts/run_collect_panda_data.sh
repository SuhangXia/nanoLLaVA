#!/bin/bash
set -e
NUM_EPISODES=1000
OUTPUT_DIR="./data/rlbench_panda_vtla"
SEED=42
MIN_EPISODE_LENGTH=20
MAX_EPISODE_LENGTH=200
GUI_FLAG=""
NO_RANDOMIZE=""

python -c "import pyrep; import rlbench" 2>/dev/null || {
    echo "ERROR: RLBench not installed."
    exit 1
}

python vtla_data_collector.py \
    --num_episodes ${NUM_EPISODES} \
    --output_dir ${OUTPUT_DIR} \
    --seed ${SEED} \
    --min_episode_length ${MIN_EPISODE_LENGTH} \
    --max_episode_length ${MAX_EPISODE_LENGTH} \
    ${GUI_FLAG} \
    ${NO_RANDOMIZE}
