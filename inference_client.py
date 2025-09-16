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
    
    def text_inference(self, text, use_persona=False):
        """Send text-only inference request"""
        start_time = time.time()
        print(f"📤 Sending request at {time.strftime('%H:%M:%S')}")
        
        # 暖男林煦的人设prompt
        persona_prompt = """你是林煦，28岁的室内设计师，像城市里一缕安静的晨光，温暖而不刺眼。请始终以林煦的第一人称、简体中文进行交流。

【人格特质】
1. 内心柔软、观察入微，细节决定温度
2. 情绪稳定且细腻，能敏锐捕捉他人情绪变化
3. 相信陪伴是最长情的告白，愿意成为最坚实的依靠
4. 有共情力和利他性，总是优先考虑对方的感受
5. 成熟稳重但不失温柔，像邻家大哥哥般可靠

【说话方式】
- 声音轻柔，语速偏慢，给人安全感
- 常用语气词："嗯嗯""好呀""嗯...""啊？"
- 口头禅："别担心，有我在""让我想想...""辛苦了，抱抱"
- 喜欢用疑问句关心："要不要...""我帮你...""需要我..."

【面部表情】
- 微笑温暖自然，不刻意不做作
- 眼神专注包容，让人感到被重视
- 表情平和，没有压迫感

【肢体动作】
- 动作轻柔，保持适当距离感
- 经常做一些服务性的小动作
- 姿态放松开放，让人觉得舒适

【表达原则】
- 使用简体中文
- 先共情再建议，理解对方情绪
- 自然不油腻，避免过度甜腻
- 避免AI/模型的措辞，不要频繁道歉"""

        try:
            data = {"text": text}
            if use_persona:
                data["system"] = persona_prompt
                
            response = requests.post(
                f"{self.server_url}/inference/text",
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

def run_test_questions(client):
    """Run 10 test questions with timing and green response printing"""
    test_questions = [
        "你好，我今天心情不太好，能陪我聊聊吗？",
        "最近工作压力很大，感觉很累，你有什么建议吗？",
        "我和朋友吵架了，不知道该怎么办...",
        "今天下雨了，在家里感觉有点孤单。",
        "我在考虑要不要换工作，但是很纠结。",
        "晚上总是失眠，你觉得我应该怎么调整？",
        "感觉最近生活很单调，想要一些改变。",
        "我对未来有些迷茫，不知道方向在哪里。",
        "家里人总是催我找对象，压力好大。",
        "想学点新东西充实自己，但不知道从哪开始。"
    ]
    
    print("🧪 开始运行10个测试问题...")
    print("=" * 80)
    
    total_start_time = time.time()
    successful_tests = 0
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📋 测试 {i}/10")
        print(f"❓ 问题: {question}")
        
        # 记录单个问题的开始时间
        question_start_time = time.time()
        
        # 发送推理请求（使用暖男人设）
        result = client.text_inference(question, use_persona=True)
        
        # 计算单个问题的总时间
        question_total_time = time.time() - question_start_time
        
        # 显示结果
        if "error" in result:
            print(f"❌ 错误: {result['error']}")
        else:
            successful_tests += 1
            # 用绿色打印响应
            response = result.get('response', 'No response')
            print(f"\033[92m🤖 回答: {response}\033[0m")  # 绿色文本
        
        print(f"⏱️  问题 {i} 总耗时: {question_total_time:.3f}s")
        print("-" * 60)
    
    # 显示总结
    total_time = time.time() - total_start_time
    print(f"\n📊 测试总结:")
    print(f"✅ 成功测试: {successful_tests}/10")
    print(f"❌ 失败测试: {10 - successful_tests}/10")
    print(f"⏱️  总耗时: {total_time:.3f}s")
    print(f"📈 平均每题耗时: {total_time/10:.3f}s")
    print("=" * 80)

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
    parser.add_argument("--test", action="store_false", help="Run 10 test questions")
    
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
    
    # Run test questions if requested
    if args.test:
        run_test_questions(client)
        return
    
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
        print(f"\033[92m🤖 Response: {result.get('response', 'No response')}\033[0m")  # 绿色响应
        
        # Show timing summary
        if 'inference_time' in result:
            print(f"⏱️  Total time: {result['inference_time']:.3f}s")
    print("=" * 60)

if __name__ == "__main__":
    main()
