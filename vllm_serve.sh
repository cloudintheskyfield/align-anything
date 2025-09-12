#!/usr/bin/env bash
#
# Copyright 2024 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Model path
MODEL_PATH="./output/qwen_omni_sft/slice_end"

# Server configuration
HOST="0.0.0.0"
PORT=10020

# GPU configuration - use only GPU 4
export CUDA_VISIBLE_DEVICES=4

echo "Starting vLLM server with model: ${MODEL_PATH}"
echo "Server will be available at: http://${HOST}:${PORT}"
echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"

# Check if model config exists and fix model_type if needed
if [ -f "${MODEL_PATH}/config.json" ]; then
    echo "Checking model configuration..."
    # Backup original config
    cp "${MODEL_PATH}/config.json" "${MODEL_PATH}/config.json.backup"
    
    # Replace qwen2_5_omni_thinker with qwen2 for compatibility
    sed -i 's/"model_type": "qwen2_5_omni_thinker"/"model_type": "qwen2"/g' "${MODEL_PATH}/config.json"
    echo "Model type updated for vLLM compatibility"
fi

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_PATH} \
    --host ${HOST} \
    --port ${PORT} \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.3 \
    --max-model-len 2048 \
    --trust-remote-code \
    --served-model-name qwen-omni-sft
