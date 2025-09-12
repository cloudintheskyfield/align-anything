#!/bin/bash
# Start inference server script

echo "🚀 Starting Qwen Omni Inference Server..."
echo "📍 Using GPU 4 for inference"
echo "🔗 Server will be available at http://localhost:8080"
echo ""

# Set environment variables
export CUDA_VISIBLE_DEVICES=4

# Start the server
python inference_server.py
