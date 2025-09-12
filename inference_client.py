#!/usr/bin/env python3
"""
Inference Client for Qwen Omni Model Server
Sends requests to the inference server for text and image processing
"""

import requests
import base64
import json
import argparse
import time
from pathlib import Path

class InferenceClient:
    def __init__(self, server_url="http://localhost:10020"):
        self.server_url = server_url.rstrip('/')
        
    def health_check(self):
        """Check if server is healthy"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def encode_image(self, image_path):
        """Encode image file to base64"""
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            raise Exception(f"Failed to encode image {image_path}: {e}")
    
    def text_inference(self, text):
        """Send text-only inference request"""
        start_time = time.time()
        print(f"📤 Sending request at {time.strftime('%H:%M:%S')}")
        try:
            response = requests.post(
                f"{self.server_url}/inference/text",
                json={"text": text},
                timeout=120
            )
            request_time = time.time() - start_time
            print(f"📥 Response received in {request_time:.3f}s")
            
            result = response.json()
            if 'inference_time' in result:
                print(f"🧠 Server inference time: {result['inference_time']:.3f}s")
            return result
        except Exception as e:
            request_time = time.time() - start_time
            print(f"❌ Request failed after {request_time:.3f}s")
            return {"error": str(e)}
    
    def multimodal_inference(self, text, image_path=None):
        """Send multimodal inference request"""
        start_time = time.time()
        print(f"📤 Sending multimodal request at {time.strftime('%H:%M:%S')}")
        if image_path:
            print(f"🖼️  Including image: {image_path}")
        
        try:
            data = {"text": text}
            
            if image_path:
                encode_start = time.time()
                data["image"] = self.encode_image(image_path)
                encode_time = time.time() - encode_start
                print(f"📷 Image encoding time: {encode_time:.3f}s")
            
            response = requests.post(
                f"{self.server_url}/inference",
                json=data,
                timeout=120
            )
            request_time = time.time() - start_time
            print(f"📥 Response received in {request_time:.3f}s")
            
            result = response.json()
            if 'inference_time' in result:
                print(f"🧠 Server inference time: {result['inference_time']:.3f}s")
            return result
        except Exception as e:
            request_time = time.time() - start_time
            print(f"❌ Request failed after {request_time:.3f}s")
            return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Qwen Omni Inference Client")
    parser.add_argument("--server", default="http://localhost:10020", help="Server URL")
    parser.add_argument(
        "--text",
        default='Hello, how are you?',
        required=False,
        help="Text input for inference"
    )
    parser.add_argument("--image", help="Path to image file (optional)")
    parser.add_argument("--check-health", action="store_true", help="Check server health")
    
    args = parser.parse_args()
    
    client = InferenceClient(args.server)
    
    # Health check if requested
    if args.check_health:
        print("🔍 Checking server health...")
        health = client.health_check()
        print(f"Health status: {json.dumps(health, indent=2)}")
        if health.get('status') != 'healthy':
            print("❌ Server is not healthy!")
            return
        print("✅ Server is healthy!")
        print()
    
    # Perform inference
    print(f"📝 Text input: {args.text}")
    if args.image:
        print(f"🖼️  Image: {args.image}")
        if not Path(args.image).exists():
            print(f"❌ Image file not found: {args.image}")
            return
        
        result = client.multimodal_inference(args.text, args.image)
    else:
        result = client.text_inference(args.text)
    
    # Display results
    print("=" * 60)
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print("✅ Inference successful!")
        print(f"🤖 Response: {result.get('response', 'No response')}")
        
        # Show timing summary
        if 'inference_time' in result:
            print(f"⏱️  Total time: {result['inference_time']:.3f}s")
    print("=" * 60)

if __name__ == "__main__":
    main()
