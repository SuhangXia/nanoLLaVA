#!/bin/bash
# Collect RLBench-Panda VTLA Expert Demonstrations
# 
# This script runs the Oracle data collector to generate HDF5 trajectories
# in ViTaMIn-B format for VLA Phase 1 training.

set -e

# Configuration
NUM_EPISODES=1000
OUTPUT_DIR="./data/rlbench_panda_vtla"
SEED=42
MIN_EPISODE_LENGTH=20
MAX_EPISODE_LENGTH=200

# Optional: Enable GUI for visualization (slower)
# GUI_FLAG="--gui"
GUI_FLAG=""

# Optional: Disable domain randomization for debugging
# NO_RANDOMIZE="--no_randomize"
NO_RANDOMIZE=""

echo "=============================================="
echo "RLBench-Panda VTLA Data Collection"
echo "=============================================="
echo "Output directory: ${OUTPUT_DIR}"
echo "Number of episodes: ${NUM_EPISODES}"
echo "Domain randomization: $([ -z "$NO_RANDOMIZE" ] && echo "Enabled" || echo "Disabled")"
echo "=============================================="

# Check RLBench installation
python -c "import pyrep; import rlbench" 2>/dev/null || {
    echo "ERROR: RLBench not installed. Install with:"
    echo "  pip install git+https://github.com/stepjam/PyRep.git"
    echo "  pip install git+https://github.com/stepjam/RLBench.git"
    exit 1
}

# Run data collector
python vtla_data_collector.py \
    --num_episodes ${NUM_EPISODES} \
    --output_dir ${OUTPUT_DIR} \
    --seed ${SEED} \
    --min_episode_length ${MIN_EPISODE_LENGTH} \
    --max_episode_length ${MAX_EPISODE_LENGTH} \
    ${GUI_FLAG} \
    ${NO_RANDOMIZE}

echo ""
echo "=============================================="
echo "Data collection complete!"
echo "Saved to: ${OUTPUT_DIR}"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Verify data: ls -lh ${OUTPUT_DIR}/*.hdf5"
echo "  2. Train VLA: bash scripts/run_train_vla_panda.sh"
echo "=============================================="
