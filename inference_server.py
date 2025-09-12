#!/usr/bin/env python3
"""
Inference Server for Qwen Omni Model
Loads model once and serves inference requests via HTTP API
"""

import os
import torch
import base64
import json
import time
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor

# Set GPU 4 for inference
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

app = Flask(__name__)

# Global variables for model and processor
model = None
processor = None

def load_model():
    """Load model and processor once at startup"""
    global model, processor
    
    model_path = '/mnt/data3/nlp/ws/proj/align-anything/output/qwen_omni_sft/slice_7'
    
    print("🚀 Loading model and processor...")
    
    # Load model
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    
    # Load processor
    processor = Qwen2_5OmniProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    print("✅ Model and processor loaded successfully!")
    print(f"📍 Model device: {model.device}")

def inference(text_input, image_data=None):
    """
    Perform inference with the loaded model
    Args:
        text_input: Text prompt
        image_data: PIL Image object or None
    """
    start_time = time.time()
    print(f"🚀 Starting inference at {time.strftime('%H:%M:%S')}")
    print(f"📝 Text input: {text_input[:100]}{'...' if len(text_input) > 100 else ''}")
    print(f"🖼️  Has image: {image_data is not None}")
    
    # Preprocessing time
    prep_start = time.time()
    if image_data:
        inputs = processor(text=text_input, images=image_data, return_tensors="pt")
    else:
        inputs = processor(text=text_input, return_tensors="pt")
    
    # Move inputs to same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prep_time = time.time() - prep_start
    print(f"⚙️  Preprocessing time: {prep_time:.3f}s")
    
    # Generation time
    gen_start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=1,
            top_p=0.8,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            early_stopping=True,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    gen_time = time.time() - gen_start
    print(f"🧠 Generation time: {gen_time:.3f}s")
    
    # Decoding time
    decode_start = time.time()
    response = processor.decode(outputs[0], skip_special_tokens=True)
    decode_time = time.time() - decode_start
    print(f"📄 Decoding time: {decode_time:.3f}s")
    
    total_time = time.time() - start_time
    print(f"✅ Total inference time: {total_time:.3f}s")
    print(f"🤖 Response length: {len(response)} chars")
    print("-" * 50)
    
    return response

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'processor_loaded': processor is not None,
        'device': str(model.device) if model else None
    })

@app.route('/inference', methods=['POST'])
def inference_endpoint():
    """
    Inference endpoint
    Expects JSON with:
    - text: string (required)
    - image: base64 encoded image (optional)
    """
    request_start = time.time()
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text input'}), 400
        
        text_input = data['text']
        image_data = None
        
        # Handle image if provided
        if 'image' in data and data['image']:
            try:
                # Decode base64 image
                image_bytes = base64.b64decode(data['image'])
                image_data = Image.open(BytesIO(image_bytes))
            except Exception as e:
                return jsonify({'error': f'Invalid image data: {str(e)}'}), 400
        
        # Perform inference
        result = inference(text_input, image_data)
        
        request_time = time.time() - request_start
        print(f"🌐 Total request time: {request_time:.3f}s")
        
        return jsonify({
            'success': True,
            'text': text_input,
            'has_image': image_data is not None,
            'response': result,
            'inference_time': request_time
        })
        
    except Exception as e:
        request_time = time.time() - request_start
        print(f"❌ Request failed after {request_time:.3f}s: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/inference/text', methods=['POST'])
def text_inference():
    """Simple text-only inference endpoint"""
    request_start = time.time()
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text input'}), 400
        
        result = inference(data['text'])
        
        request_time = time.time() - request_start
        print(f"🌐 Total text request time: {request_time:.3f}s")
        
        return jsonify({
            'success': True,
            'response': result,
            'inference_time': request_time
        })
        
    except Exception as e:
        request_time = time.time() - request_start
        print(f"❌ Text request failed after {request_time:.3f}s: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load model at startup
    load_model()
    
    print("🌐 Starting inference server...")
    print("📡 Endpoints:")
    print("  - GET  /health - Health check")
    print("  - POST /inference - Full inference (text + optional image)")
    print("  - POST /inference/text - Text-only inference")
    print("🔗 Server running on http://localhost:10020")
    print("⏹️  Press Ctrl+C to stop")
    
    app.run(host='0.0.0.0', port=10020, debug=False)
