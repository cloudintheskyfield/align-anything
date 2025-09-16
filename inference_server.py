#!/usr/bin/env python3
"""
Inference Server for Qwen Omni Model
Loads model once and serves inference requests via HTTP API
"""

import os

# Set GPUs 4,5 for multi-GPU inference
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

import torch
import base64
import json
import time
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify, Response
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
from qwen_omni_utils import process_mm_info



app = Flask(__name__)

# Global variables for model and processor
model = None
processor = None

def load_model():
    """Load model and processor once at startup"""
    global model, processor
    
    model_path = '/mnt/data3/nlp/ws/proj/align-anything/output/qwen_omni_sft/slice_1500'
    
    print("🚀 Loading model and processor...")
    
    # Load model with proper multi-GPU device mapping
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,  # Use dtype instead of torch_dtype
        device_map="balanced",  # Automatically balance across available GPUs
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        max_memory={0: "70GB", 1: "70GB"}  # Limit memory per GPU to avoid OOM
    )
    
    # Enable optimizations
    model.eval()
    
    # Load processor
    processor = Qwen2_5OmniProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    print("✅ Model and processor loaded successfully!")
    print(f"🧠 Model dtype: {model.dtype}")
    print(f"🔥 CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"🎯 Available GPUs: {torch.cuda.device_count()}")
        print(f"💾 Current device: {torch.cuda.current_device()}")
        
        # Show device mapping for multi-GPU
        if hasattr(model, 'hf_device_map'):
            print("🗺️  Model device mapping:")
            for module, device in model.hf_device_map.items():
                print(f"   {module}: {device}")
        
        # Show GPU memory for each available GPU
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"💾 GPU {i}: {props.name} - {props.total_memory / 1024**3:.1f}GB")

def inference_streaming(conversation):
    """
    Perform streaming inference with the loaded model using official Qwen format
    Args:
        conversation: List of message dictionaries in Qwen format
    Yields:
        dict: Streaming response chunks with token and metadata
    """
    start_time = time.time()
    print(f"🚀 Starting streaming inference at {time.strftime('%H:%M:%S')}")
    print(f"📝 Conversation: {len(conversation)} messages")
    
    # Preprocessing time - use official format
    prep_start = time.time()
    
    # Apply chat template
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    print(f"📝 Generated text: {text[:200]}{'...' if len(text) > 200 else ''}")
    
    # Process multimedia info
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    
    # Prepare inputs
    inputs = processor(
        text=text, 
        audio=audios, 
        images=images, 
        videos=videos, 
        return_tensors="pt", 
        padding=True
    )
    
    # Move inputs to appropriate devices for multi-GPU setup
    first_device = model.device if hasattr(model, 'device') else next(model.parameters()).device
    inputs = {k: v.to(first_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    # Convert to model dtype for tensor inputs
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor) and v.dtype != torch.long:  # Don't convert input_ids
            inputs[k] = v.to(model.dtype)
    prep_time = time.time() - prep_start
    print(f"⚙️  Preprocessing time: {prep_time:.3f}s")
    
    # Streaming generation
    gen_start = time.time()
    print(f"🎯 Running streaming inference on device: {first_device}")
    
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = []
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        # Generate tokens one by one for streaming
        for step in range(2048):  # Reduce max_new_tokens for faster streaming
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=True,
                temperature=0.8,
                top_p=0.8,
                repetition_penalty=1.05,
                pad_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1
            )
            
            # Handle different output formats
            if isinstance(outputs, tuple) and len(outputs) == 2:
                text_ids, audio = outputs
            else:
                text_ids = outputs
                audio = None
            
            # Get new token
            new_token_id = text_ids[0, -1].item()
            
            # Check for EOS token
            if new_token_id == processor.tokenizer.eos_token_id:
                break
            
            # Decode new token
            new_token = processor.tokenizer.decode([new_token_id], skip_special_tokens=True)
            generated_tokens.append(new_token)
            
            # Yield streaming response
            yield {
                'token': new_token,
                'partial_response': ''.join(generated_tokens),
                'step': step + 1,
                'finished': False
            }
            
            # Update inputs for next iteration
            inputs['input_ids'] = text_ids
            if 'attention_mask' in inputs:
                attention_mask = torch.cat([
                    inputs['attention_mask'], 
                    torch.ones((1, 1), device=inputs['attention_mask'].device, dtype=inputs['attention_mask'].dtype)
                ], dim=1)
                inputs['attention_mask'] = attention_mask
    
    gen_time = time.time() - gen_start
    final_response = ''.join(generated_tokens)
    
    print(f"🧠 Streaming generation time: {gen_time:.3f}s")
    print(f"🤖 Final response length: {len(final_response)} chars")
    print("-" * 50)
    
    # Final response
    yield {
        'token': '',
        'partial_response': final_response,
        'step': len(generated_tokens),
        'finished': True,
        'total_time': time.time() - start_time,
        'generation_time': gen_time
    }

def inference(conversation):
    """
    Perform inference with the loaded model using official Qwen format
    Args:
        conversation: List of message dictionaries in Qwen format
    """
    start_time = time.time()
    print(f"🚀 Starting inference at {time.strftime('%H:%M:%S')}")
    print(f"📝 Conversation: {len(conversation)} messages")
    
    # Preprocessing time - use official format
    prep_start = time.time()
    
    # Apply chat template
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    print(f"📝 Generated text: {text[:200]}{'...' if len(text) > 200 else ''}")
    
    # Process multimedia info
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    
    # Prepare inputs
    inputs = processor(
        text=text, 
        audio=audios, 
        images=images, 
        videos=videos, 
        return_tensors="pt", 
        padding=True
    )
    
    # Move inputs to appropriate devices for multi-GPU setup
    # For multi-GPU models, inputs should go to the device of the first layer
    first_device = model.device if hasattr(model, 'device') else next(model.parameters()).device
    inputs = {k: v.to(first_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    # Convert to model dtype for tensor inputs
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor) and v.dtype != torch.long:  # Don't convert input_ids
            inputs[k] = v.to(model.dtype)
    prep_time = time.time() - prep_start
    print(f"⚙️  Preprocessing time: {prep_time:.3f}s")
    
    # Generation time with optimizations
    gen_start = time.time()
    print(f"🎯 Running inference on device: {first_device}")
    with torch.no_grad(), torch.cuda.amp.autocast():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,  # Reduce from 128 to 64 for faster generation
            do_sample=True,
            temperature=0.8,  # Lower temperature for more focused generation
            top_p=0.8,  # Slightly lower top_p
            repetition_penalty=1.05,  # Reduce repetition penalty
            early_stopping=True,
            pad_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1  # Use greedy decoding for speed
        )
        
        # Handle different output formats
        if isinstance(outputs, tuple) and len(outputs) == 2:
            text_ids, audio = outputs
        else:
            text_ids = outputs
            audio = None
    gen_time = time.time() - gen_start
    print(f"🧠 Generation time: {gen_time:.3f}s")
    
    # Decoding time
    decode_start = time.time()
    response = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
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

@app.route('/inference/stream', methods=['POST'])
def inference_stream_endpoint():
    """
    Streaming inference endpoint using Server-Sent Events (SSE)
    Expects JSON with:
    - conversation: list of message dictionaries (required)
    OR
    - text: string (required) - will be converted to conversation format
    - system: string (optional) - system prompt
    - image: base64 encoded image (optional)
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Missing request data'}), 400
        
        # Handle conversation format or convert from text
        if 'conversation' in data:
            conversation = data['conversation']
        elif 'text' in data:
            # Convert text input to conversation format
            conversation = []
            
            # Add system message if provided
            if 'system' in data and data['system']:
                conversation.append({
                    "role": "system",
                    "content": [{"type": "text", "text": data['system']}]
                })
            
            # Add user message
            user_content = []
            
            # Handle image if provided
            if 'image' in data and data['image']:
                try:
                    # Decode base64 image
                    image_bytes = base64.b64decode(data['image'])
                    image_data = Image.open(BytesIO(image_bytes))
                    user_content.append({"type": "image", "image": image_data})
                except Exception as e:
                    return jsonify({'error': f'Invalid image data: {str(e)}'}), 400
            
            user_content.append({"type": "text", "text": data['text']})
            conversation.append({
                "role": "user",
                "content": user_content
            })
        else:
            return jsonify({'error': 'Missing conversation or text input'}), 400
        
        def generate_stream():
            """Generator function for streaming response"""
            try:
                for chunk in inference_streaming(conversation):
                    # Format as Server-Sent Events
                    yield f"data: {json.dumps(chunk)}\n\n"
            except Exception as e:
                error_chunk = {
                    'error': str(e),
                    'finished': True
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
        
        return Response(
            generate_stream(),
            mimetype='text/plain',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*'
            }
        )
        
    except Exception as e:
        print(f"❌ Streaming request failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/inference', methods=['POST'])
def inference_endpoint():
    """
    Inference endpoint using official Qwen conversation format
    Expects JSON with:
    - conversation: list of message dictionaries (required)
    OR
    - text: string (required) - will be converted to conversation format
    - system: string (optional) - system prompt
    - image: base64 encoded image (optional)
    """
    request_start = time.time()
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Missing request data'}), 400
        
        # Handle conversation format or convert from text
        if 'conversation' in data:
            conversation = data['conversation']
        elif 'text' in data:
            # Convert text input to conversation format
            conversation = []
            
            # Add system message if provided
            if 'system' in data and data['system']:
                conversation.append({
                    "role": "system",
                    "content": [{"type": "text", "text": data['system']}]
                })
            
            # Add user message
            user_content = []
            
            # Handle image if provided
            if 'image' in data and data['image']:
                try:
                    # Decode base64 image
                    image_bytes = base64.b64decode(data['image'])
                    image_data = Image.open(BytesIO(image_bytes))
                    user_content.append({"type": "image", "image": image_data})
                except Exception as e:
                    return jsonify({'error': f'Invalid image data: {str(e)}'}), 400
            
            user_content.append({"type": "text", "text": data['text']})
            conversation.append({
                "role": "user",
                "content": user_content
            })
        else:
            return jsonify({'error': 'Missing conversation or text input'}), 400
        
        # Perform inference
        result = inference(conversation)
        
        request_time = time.time() - request_start
        print(f"🌐 Total request time: {request_time:.3f}s")
        
        return jsonify({
            'success': True,
            'response': result,
            'inference_time': request_time
        })
        
    except Exception as e:
        request_time = time.time() - request_start
        print(f"❌ Request failed after {request_time:.3f}s: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/inference/text/stream', methods=['POST'])
def text_inference_stream():
    """Streaming text-only inference endpoint using Server-Sent Events (SSE)"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text input'}), 400
        
        # Convert to conversation format
        conversation = []
        
        # Add system message if provided
        if 'system' in data and data['system']:
            conversation.append({
                "role": "system",
                "content": [{"type": "text", "text": data['system']}]
            })
        
        # Add user message
        conversation.append({
            "role": "user",
            "content": [{"type": "text", "text": data['text']}]
        })
        
        def generate_stream():
            """Generator function for streaming text response"""
            try:
                for chunk in inference_streaming(conversation):
                    # Format as Server-Sent Events
                    yield f"data: {json.dumps(chunk)}\n\n"
            except Exception as e:
                error_chunk = {
                    'error': str(e),
                    'finished': True
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
        
        return Response(
            generate_stream(),
            mimetype='text/plain',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*'
            }
        )
        
    except Exception as e:
        print(f"❌ Streaming text request failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/inference/text', methods=['POST'])
def text_inference():
    """Simple text-only inference endpoint using official Qwen conversation format"""
    request_start = time.time()
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text input'}), 400
        
        # Convert to conversation format
        conversation = []
        
        # Add system message if provided
        if 'system' in data and data['system']:
            conversation.append({
                "role": "system",
                "content": [{"type": "text", "text": data['system']}]
            })
        
        # Add user message
        conversation.append({
            "role": "user",
            "content": [{"type": "text", "text": data['text']}]
        })
        
        result = inference(conversation)
        
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
    print("  - POST /inference/stream - Streaming full inference")
    print("  - POST /inference/text - Text-only inference")
    print("  - POST /inference/text/stream - Streaming text-only inference")
    print("🔗 Server running on http://localhost:10020")
    print("⏹️  Press Ctrl+C to stop")
    
    app.run(host='0.0.0.0', port=10020, debug=False)
