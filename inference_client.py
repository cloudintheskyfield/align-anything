#!/usr/bin/env python3
"""
Inference Client for Qwen Omni Model Server
Sends requests to the inference server for text and image processing

=== 使用示例 / Usage Examples ===

1. 基本文本推理 / Basic Text Inference:
   python inference_client.py --text "你好，你是谁？"
   python inference_client.py --text "Hello, how are you?"

2. 使用vLLM服务器推理 / Using vLLM Server:
   python inference_client.py --use-vllm --text "我最近工作压力很大，你能给我一些建议吗？"
   python inference_client.py --use-vllm --vllm-url http://127.0.0.1:10011/v1/chat/completions --text "请介绍一下Python"

3. 多模态推理（文本+图片）/ Multimodal Inference:
   python inference_client.py --text "请描述这张图片" --image ./data/test_image_1.jpg
   python inference_client.py --text "这张图片给你什么感觉？" --image /path/to/your/image.jpg

4. 服务器健康检查 / Health Check:
   python inference_client.py --check-health
   python inference_client.py --check-health --server http://localhost:10020

5. 运行测试问题 / Run Test Questions:
   python inference_client.py --test                    # 使用本地推理服务器
   python inference_client.py --test --use-vllm         # 使用vLLM服务器

6. 完整评估测试 / Complete Assessment:
   python inference_client.py --assessment                                           # 默认输出到 data/assessment_<timestamp>.parquet
   python inference_client.py --assessment --assessment-output ./data/my_eval.parquet # 自定义输出文件
   python inference_client.py --assessment --use-vllm                                # 使用vLLM进行评估
   python inference_client.py --assessment --use-vllm --assessment-output ./data/vllm_eval.parquet

7. 图像多模态测试 / Image Assessment:
   python inference_client.py --image-test                                          # 默认输出到 data/image_assessment_<timestamp>.parquet
   python inference_client.py --image-test --image-output ./data/img_test.parquet   # 自定义输出文件

8. 组合使用 / Combined Usage:
   python inference_client.py --check-health --server http://localhost:10020
   python inference_client.py --assessment --use-vllm --assessment-output ./data/runs/vllm_$(date +%Y%m%d).parquet
   python inference_client.py --text "请用温暖的语言安慰我" --use-vllm

=== 服务器配置 / Server Configuration ===

默认服务器地址 / Default Server URLs:
- 本地推理服务器 / Local Inference Server: http://localhost:10020
- vLLM服务器 / vLLM Server: http://127.0.0.1:10011/v1/chat/completions  
- LLaMA4评分服务器 / LLaMA4 Scoring Server: http://127.0.0.1:10018/v1/chat/completions

模型配置 / Model Configuration:
- 本地模型 / Local Model: Qwen2_5OmniThinkerForConditionalGeneration
- vLLM模型 / vLLM Model: /mnt/data3/nlp/ws/model/Qwen2/Qwen/Qwen2.5-Omni-7B

=== 输出文件 / Output Files ===

评估结果文件 / Assessment Output Files:
- assessment_<timestamp>.parquet: 评估数据（包含有人设和无人设测试）
- assessment_<timestamp>_metadata.json: 数据字典和统计信息
- image_assessment_<timestamp>.parquet: 图像测试数据
- image_assessment_<timestamp>_metadata.json: 图像测试元数据

=== 环境要求 / Requirements ===

Python包依赖 / Python Dependencies:
- requests, pandas, pyarrow (or fastparquet), pathlib, argparse

服务器要求 / Server Requirements:
- 本地推理服务器运行在端口10020 / Local inference server on port 10020
- vLLM服务器运行在端口10011 / vLLM server on port 10011  
- LLaMA4评分服务器运行在端口10018 / LLaMA4 scoring server on port 10018

=== 注意事项 / Notes ===

1. 使用--use-vllm时会连接到vLLM部署的Qwen2.5-Omni-7B模型
2. 评估测试包含暖男人设和无人设两组对比测试
3. 所有测试结果都会保存为parquet格式，便于后续分析
4. 图像测试需要在data/目录下有test_image_*.jpg文件
5. 评分使用LLaMA4模型进行1-10分的人设符合度评估

"""
import os

import requests
import base64
import json
import argparse
import time
import re
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path
from transformers import Qwen2_5OmniThinkerForConditionalGeneration

class InferenceClient:
    def __init__(self, server_url="http://localhost:10020", llama4_url="http://127.0.0.1:10018/v1/chat/completions", vllm_url="http://127.0.0.1:10011/v1/chat/completions"):
        self.server_url = server_url.rstrip('/')
        self.llama4_url = llama4_url
        self.vllm_url = vllm_url
        self.session = requests.Session()
        self.use_streaming = True  # Enable streaming by default
        self._vllm_model_cache = None  # Cache the detected model name
        
    def health_check(self):
        """Check if server is healthy"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_vllm_model_name(self):
        """Auto-detect the correct model name from vLLM server"""
        if self._vllm_model_cache:
            return self._vllm_model_cache
            
        # Check environment variable first
        env_model = os.getenv("VLLM_MODEL")
        if env_model:
            self._vllm_model_cache = env_model
            return env_model
            
        try:
            # Query vLLM server for available models
            models_url = self.vllm_url.replace('/v1/chat/completions', '/v1/models')
            response = requests.get(models_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and data['data']:
                    model_name = data['data'][0]['id']
                    print(f"🔍 Auto-detected vLLM model: {model_name}")
                    self._vllm_model_cache = model_name
                    return model_name
        except Exception as e:
            print(f"⚠️  Failed to auto-detect model: {e}")
            
        # Fallback to common possibilities
        fallback_models = [
            "Qwen2___5-Omni-7B",  # directory name (most likely)
            "Qwen2.5-Omni-7B",   # served model name
            "/mnt/data3/nlp/ws/model/Qwen2/Qwen/Qwen2___5-Omni-7B"  # full path
        ]
        
        for model in fallback_models:
            print(f"🔄 Trying model name: {model}")
            # Quick test with a minimal request
            try:
                test_payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 1,
                    "stream": False
                }
                test_resp = requests.post(self.vllm_url, json=test_payload, timeout=10)
                if test_resp.status_code != 404:  # 404 means model not found
                    print(f"✅ Found working model name: {model}")
                    self._vllm_model_cache = model
                    return model
            except Exception:
                continue
                
        # Final fallback
        default_model = "Qwen2___5-Omni-7B"
        print(f"⚠️  Using default model name: {default_model}")
        self._vllm_model_cache = default_model
        return default_model

    def encode_image(self, image_path):
        """Encode image file to base64"""
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            raise Exception(f"Failed to encode image {image_path}: {e}")
    
    def text_inference_streaming(self, text, system_prompt=None, use_persona=False):
        """Send streaming text inference request to server"""
        start_time = time.time()
        
        # Use warm-hearted persona if requested
        if use_persona:
            system_prompt = """你叫林煦，是一位28岁的室内设计师。你像城市里一缕安静的晨光，温暖而不刺眼。

【人格特点】
- 内心柔软细腻，情绪稳定，有共情力与利他性
- 注重细节与陪伴，善于倾听和理解他人

【说话方式】
- 声音轻、语速慢，给人安全感
- 常用语气词：'嗯嗯''好呀''嗯...''啊？'
- 口头禅：'别担心，有我在''让我想想...''辛苦了，抱抱'
- 喜欢用疑问句关心：'要不要...''我帮你...'

【面部表情】
- 微笑温暖自然，不做作
- 眼神专注包容，让人感到被理解
- 表情平和，没有压迫感

【肢体动作】
- 动作轻柔，保持适当距离感
- 有服务性的小动作（递纸巾、倒水等）
- 姿态放松开放，不紧张

【表达原则】
- 使用简体中文
- 先共情再建议，避免直接说教
- 表达自然不油腻，避免过度甜腻
- 避免AI/模型常用措辞，避免道歉模板"""
        
        print(f"📤 Sending streaming request at {time.strftime('%H:%M:%S')}")
        
        try:
            payload = {
                "text": text
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            response = self.session.post(
                f"{self.server_url}/inference/text/stream",
                json=payload,
                timeout=60,
                stream=True
            )
            
            full_response = ""
            print(f"🤖 回答: ", end='', flush=True)
            print("\033[92m", end='', flush=True)  # Start green color
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        try:
                            data = json.loads(line[6:])  # Remove 'data: ' prefix
                            
                            if 'error' in data:
                                print(f"\033[0m\n❌ Streaming error: {data['error']}")
                                return {"error": data['error']}
                            
                            if 'token' in data and data['token']:
                                print(data['token'], end='', flush=True)
                            
                            if data.get('finished', False):
                                full_response = data.get('partial_response', '')
                                total_time = data.get('total_time', time.time() - start_time)
                                gen_time = data.get('generation_time', 0)
                                
                                print("\033[0m")  # Reset color
                                print(f"📥 Response received in {total_time:.3f}s")
                                print(f"🧠 Server inference time: {gen_time:.3f}s")
                                
                                return {
                                    'success': True,
                                    'response': full_response,
                                    'inference_time': total_time
                                }
                        except json.JSONDecodeError:
                            continue
            
            print("\033[0m")  # Reset color
            return {"error": "Streaming ended without completion"}
            
        except Exception as e:
            print("\033[0m")  # Reset color
            request_time = time.time() - start_time
            print(f"\n❌ Streaming request failed after {request_time:.3f}s")
            return {"error": str(e)}

    def text_inference(self, text, system_prompt=None, use_persona=False):
        """Send text inference request to server (with optional streaming)"""
        if self.use_streaming:
            return self.text_inference_streaming(text, system_prompt, use_persona)
        
        start_time = time.time()
        
        # Use warm-hearted persona if requested
        if use_persona:
            system_prompt = """你叫林煦，是一位28岁的室内设计师。你像城市里一缕安静的晨光，温暖而不刺眼。

【人格特点】
- 内心柔软细腻，情绪稳定，有共情力与利他性
- 注重细节与陪伴，善于倾听和理解他人

【说话方式】
- 声音轻、语速慢，给人安全感
- 常用语气词：'嗯嗯''好呀''嗯...''啊？'
- 口头禅：'别担心，有我在''让我想想...''辛苦了，抱抱'
- 喜欢用疑问句关心：'要不要...''我帮你...'

【面部表情】
- 微笑温暖自然，不做作
- 眼神专注包容，让人感到被理解
- 表情平和，没有压迫感

【肢体动作】
- 动作轻柔，保持适当距离感
- 有服务性的小动作（递纸巾、倒水等）
- 姿态放松开放，不紧张

【表达原则】
- 使用简体中文
- 先共情再建议，避免直接说教
- 表达自然不油腻，避免过度甜腻
- 避免AI/模型常用措辞，避免道歉模板"""
        
        print(f"📤 Sending request at {time.strftime('%H:%M:%S')}")
        
        try:
            payload = {
                "text": text
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            response = self.session.post(
                f"{self.server_url}/inference/text",
                json=payload,
                timeout=30
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
    
    def multimodal_inference_conversation(self, conversation, image_path=None):
        """Send multimodal inference request with conversation format"""
        start_time = time.time()
        print(f"📤 Sending multimodal conversation request at {time.strftime('%H:%M:%S')}")
        if image_path:
            print(f"🖼️  Including image: {image_path}")
        
        try:
            data = {"conversation": conversation}
            
            if image_path:
                encode_start = time.time()
                # Add image to user message content as URL
                for message in conversation:
                    if message["role"] == "user":
                        message["content"].append({
                            "type": "image", 
                            "image": f"{os.path.abspath(image_path)}"
                        })
                        break
                encode_time = time.time() - encode_start
                print(f"📷 Image URL processing time: {encode_time:.3f}s")
            
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
    
    def extract_assistant_response(self, response):
        """Extract only the assistant's response content, removing system/user prefixes"""
        if not response:
            return response
        
        # Split by common conversation markers
        lines = response.split('\n')
        assistant_content = []
        in_assistant_section = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('assistant'):
                in_assistant_section = True
                # Skip the "assistant" line itself
                continue
            elif line.startswith(('system', 'user')) and in_assistant_section:
                # Stop when we hit another section
                break
            elif in_assistant_section:
                assistant_content.append(line)
        
        if assistant_content:
            return '\n'.join(assistant_content).strip()
        
        # Fallback: if no "assistant" marker found, return original
        return response
    
    def stream_print_response(self, text, delay=0.02):
        """Print text with streaming effect"""
        if not text:
            return
        
        # Extract only assistant content for display
        clean_text = self.extract_assistant_response(text)
        
        print("\033[92m", end='', flush=True)  # Start green color
        for char in clean_text:
            print(char, end='', flush=True)
            time.sleep(delay)
        print("\033[0m", end='', flush=True)  # Reset color
    
    def vllm_inference(self, text, system_prompt=None, use_persona=False):
        """Send streaming inference request to vLLM server (Qwen2.5-Omni-7B)"""
        start_time = time.time()
        
        # Use warm-hearted persona if requested
        if use_persona:
            system_prompt = """你叫林煦，是一位28岁的室内设计师。你像城市里一缕安静的晨光，温暖而不刺眼。

【人格特点】
- 内心柔软细腻，情绪稳定，有共情力与利他性
- 注重细节与陪伴，善于倾听和理解他人

【说话方式】
- 声音轻、语速慢，给人安全感
- 常用语气词：'嗯嗯''好呀''嗯...''啊？'
- 口头禅：'别担心，有我在''让我想想...''辛苦了，抱抱'
- 喜欢用疑问句关心：'要不要...''我帮你...'

【面部表情】
- 微笑温暖自然，不做作
- 眼神专注包容，让人感到被理解
- 表情平和，没有压迫感

【肢体动作】
- 动作轻柔，保持适当距离感
- 有服务性的小动作（递纸巾、倒水等）
- 姿态放松开放，不紧张

【表达原则】
- 使用简体中文
- 先共情再建议，避免直接说教
- 表达自然不油腻，避免过度甜腻
- 避免AI/模型常用措辞，避免道歉模板"""
        
        print(f"📤 Sending vLLM streaming request at {time.strftime('%H:%M:%S')}")
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": text})
            
            # Auto-detect the correct model name
            vllm_model = self.get_vllm_model_name()
            payload = {
                "model": vllm_model,
                "messages": messages,
                "temperature": 0.8,
                "max_tokens": 1024,
                "top_p": 0.8,
                "stream": True
            }
            
            response = self.session.post(
                self.vllm_url,
                json=payload,
                # increase read timeout to allow for model warmup / first token latency
                # use (connect_timeout, read_timeout)
                timeout=(10, 600),
                stream=True,
                headers={
                    'Content-Type': 'application/json',
                    'Accept': 'text/event-stream'
                }
            )

            # If the server did not accept the request, surface the error body
            if response.status_code != 200:
                try:
                    err_json = response.json()
                    err_msg = err_json.get('error') or err_json
                except Exception:
                    err_msg = response.text
                print(f"\n❌ vLLM HTTP {response.status_code}: {err_msg}")
                return {"error": f"vLLM HTTP {response.status_code}", "detail": err_msg}

            # When not streaming back as SSE (e.g. JSON error), show it explicitly
            ctype = response.headers.get('content-type', '')
            if 'text/event-stream' not in ctype:
                try:
                    data = response.json()
                    return {"error": "vLLM non-SSE response", "detail": data}
                except Exception:
                    body = response.text[:2000]
                    return {"error": "vLLM non-SSE response", "detail": body}
            
            full_response = ""
            print(f"🤖 回答: ", end='', flush=True)
            print("\033[92m", end='', flush=True)  # Start green color
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        try:
                            data_str = line[6:]  # Remove 'data: ' prefix
                            if data_str.strip() == '[DONE]':
                                break
                            data = json.loads(data_str)
                            
                            if 'choices' in data and data['choices']:
                                delta = data['choices'][0].get('delta', {})
                                if 'content' in delta and delta['content']:
                                    token = delta['content']
                                    print(token, end='', flush=True)
                                    full_response += token
                        except json.JSONDecodeError:
                            continue
            
            print("\033[0m")  # Reset color
            request_time = time.time() - start_time
            print(f"\n📥 vLLM streaming completed in {request_time:.3f}s")
            
            if full_response:
                return {
                    'success': True,
                    'response': full_response,
                    'inference_time': request_time,
                    'model': 'Qwen2.5-Omni-7B-vLLM-Stream'
                }
            else:
                return {"error": "No response from vLLM server"}
                
        except Exception as e:
            request_time = time.time() - start_time
            print(f"\n❌ vLLM streaming request failed after {request_time:.3f}s: {str(e)}")
            # Fallback to non-streaming with a longer timeout to handle slow first-token scenarios
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": text})
                # Use auto-detected model name in fallback
                vllm_model = self.get_vllm_model_name()
                payload = {
                    "model": vllm_model,
                    "messages": messages,
                    "temperature": 0.8,
                    "max_tokens": 1024,
                    "top_p": 0.8,
                    "stream": False
                }
                print("🔁 Retrying with non-streaming mode (extended timeout)...")
                start_fallback = time.time()
                resp = self.session.post(
                    self.vllm_url,
                    json=payload,
                    timeout=(10, 600)
                )
                resp.raise_for_status()
                data = resp.json()
                if 'choices' in data and data['choices']:
                    content = data['choices'][0].get('message', {}).get('content', '')
                    fb_time = time.time() - start_fallback
                    return {
                        'success': True,
                        'response': content,
                        'inference_time': fb_time,
                        'model': 'Qwen2.5-Omni-7B-vLLM-NonStream'
                    }
                return {"error": "No response from vLLM server (non-streaming)"}
            except Exception as e2:
                return {"error": f"vLLM request failed (streaming and non-streaming): {e2}"}
    
    def score_response_with_llama4(self, prompt, response):
        """Use llama4 model to score response based on warm-hearted persona (1-10)"""
        if not response or response == "No response":
            return 0.0, "无有效回复"
        
        # Extract only assistant response content for scoring
        clean_response = self.extract_assistant_response(response)
        
        persona_rubric = """你叫林煦，是一位28岁的室内设计师。你像城市里一缕安静的晨光，温暖而不刺眼。
【人格特点】内心柔软细腻、情绪稳定、有共情力与利他性；注重细节与陪伴。
【说话方式】声音轻、语速慢；常用'嗯嗯''好呀''嗯...''啊？'；口头禅'别担心，有我在''让我想想...''辛苦了，抱抱'；喜欢用'要不要...''我帮你...'。
【面部表情】微笑温暖自然；眼神专注包容；表情平和无压迫感。
【肢体动作】动作轻柔、保持距离感、服务性小动作、姿态放松开放。
【表达原则】简体中文、先共情再建议、自然不油腻、避免AI/模型措辞与道歉模板。"""
        
        judge_messages = [
            {
                "role": "system",
                "content": "你是严格的对话质量评估员，依据给定'暖男-林煦'人设与规范，对回复进行1-10分打分并给出详细评分理由。格式：分数|理由"
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"人设规范：\n{persona_rubric}"},
                    {"type": "text", "text": f"用户问题：\n{prompt}"},
                    {"type": "text", "text": f"候选回复：\n{clean_response}"},
                    {"type": "text", "text": "请根据人设、说话方式、共情与建议的到位程度、自然度、无模板化、无AI措辞等维度，给出1-10分评分。格式：分数|详细理由（包括符合/不符合人设的具体表现）"}
                ]
            }
        ]
        
        payload = {
            "messages": judge_messages,
            "model": "/mnt/data3/nlp/ws/model/llama_4_maverick",
            "temperature": 0.1,
            "max_tokens": 1024,
            "top_p": 0.5,
            "stream": False,
            "top_k": 1,
            "min_p": 0.1,
            "use_beam_search": False,
            "repetition_penalty": 1.0,
            "logprobs": False,
            "skip_special_tokens": True,
            "echo": False
        }
        
        try:
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            resp = self.session.post(
                self.llama4_url, 
                json=payload, 
                headers=headers, 
                timeout=60, 
                verify=False
            )
            resp.raise_for_status()
            
            data = resp.json()
            text = ""
            if 'choices' in data and data['choices']:
                text = data['choices'][0]['message']['content'] or ""
            
            # 解析分数和理由
            if '|' in text:
                parts = text.split('|', 1)
                score_text = parts[0].strip()
                reason = parts[1].strip()
            else:
                score_text = text.strip()
                reason = "无详细理由"
            
            # 提取分数
            score_match = re.search(r"(10(?:\.0)?|[0-9](?:\.[0-9])?)", score_text)
            if score_match:
                score = float(score_match.group(1))
                score = max(0.0, min(10.0, score))  # 限制在0-10范围
            else:
                score = 0.0
                reason = f"无法解析分数: {text}"
            
            return score, reason
            
        except Exception as e:
            print(f"❌ 评分失败: {e}")
            return 0.0, f"评分API调用失败: {str(e)}"

def run_test_questions(use_vllm=False):
    """Run test questions with warm-hearted persona"""
    client = InferenceClient()
    
    # Check server health
    health = client.health_check()
    if "error" in health:
        print(f"❌ Server health check failed: {health['error']}")
        return
    
    print("✅ Server is healthy")
    print(f"🤖 Model loaded: {health.get('model_loaded', False)}")
    print(f"🔧 Processor loaded: {health.get('processor_loaded', False)}")
    print(f"🎯 Device: {health.get('device', 'Unknown')}")
    
    # Test questions with warm-hearted persona
    test_questions = [
        "我最近工作压力很大，经常加班到很晚，感觉身心俱疲，你能给我一些建议吗？",
        "我和男朋友吵架了，他说我太敏感，但我觉得他不理解我的感受，我该怎么办？",
        "我刚搬到新城市，人生地不熟的，感觉很孤独，有什么方法能快速适应新环境吗？",
        "我妈妈总是催我结婚，但我现在还不想定下来，每次回家都很有压力，怎么处理这种情况？",
        "我的好朋友最近总是向我抱怨她的生活，我想帮助她但又不知道该说什么，感觉很无力。",
        "我在考虑要不要辞职去追求自己的梦想，但又担心经济压力，内心很纠结。",
        "我发现自己越来越容易焦虑，特别是面对不确定的事情时，有什么方法能缓解吗？",
        "我和室友的生活习惯差异很大，经常因为小事产生摩擦，但又不想破坏关系。",
        "我最近失眠很严重，晚上总是胡思乱想睡不着，白天又没精神，该怎么调整？",
        "我觉得自己在朋友圈里总是那个倾听者，但当我需要倾诉时却找不到合适的人。"
    ]
    
    print(f"\n🚀 开始测试 {len(test_questions)} 个问题 (使用暖男人设)")
    print("=" * 80)
    
    successful_tests = 0
    total_start_time = time.time()
    all_scores = []  # 收集所有评分
    assessment_data = []  # 收集评估数据
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 测试 {i}/10")
        print(f"❓ 问题: {question}")
        
        question_start_time = time.time()
        
        # 发送推理请求
        if use_vllm:
            result = client.vllm_inference(question, use_persona=True)
        else:
            result = client.text_inference(question, use_persona=True)
        
        # 计算单个问题的总时间
        question_total_time = time.time() - question_start_time
        
        # 显示结果
        if "error" in result:
            print(f"❌ 错误: {result['error']}")
            # 记录错误情况
            assessment_data.append({
                'timestamp': datetime.now().isoformat(),
                'question_id': i,
                'question': question,
                'response': None,
                'clean_response': None,
                'score': 0.0,
                'reason': f"推理错误: {result['error']}",
                'inference_time': question_total_time,
                'use_persona': True,
                'test_type': 'with_persona',
                'system_prompt': """你叫林煦，是一位28岁的室内设计师。你像城市里一缕安静的晨光，温暖而不刺眼。

【人格特点】
- 内心柔软细腻，情绪稳定，有共情力与利他性
- 注重细节与陪伴，善于倾听和理解他人

【说话方式】
- 声音轻、语速慢，给人安全感
- 常用语气词：'嗯嗯''好呀''嗯...''啊？'
- 口头禅：'别担心，有我在''让我想想...''辛苦了，抱抱'
- 喜欢用疑问句关心：'要不要...''我帮你...'

【面部表情】
- 微笑温暖自然，不做作
- 眼神专注包容，让人感到被理解
- 表情平和，没有压迫感

【肢体动作】
- 动作轻柔，保持适当距离感
- 有服务性的小动作（递纸巾、倒水等）
- 姿态放松开放，不紧张

【表达原则】
- 使用简体中文
- 先共情再建议，避免直接说教
- 表达自然不油腻，避免过度甜腻
- 避免AI/模型常用措辞，避免道歉模板""",
                'model_name': 'Qwen2_5OmniThinkerForConditionalGeneration',
                'server_url': client.server_url,
                'llama4_url': client.llama4_url
            })
        else:
            successful_tests += 1
            # 响应已在streaming中显示，无需重复打印
            response = result.get('response', 'No response')
            
            # 使用llama4进行评分
            print("🔍 正在评分...")
            score, reason = client.score_response_with_llama4(question, response)
            all_scores.append(score)
            # 用紫色打印评分结果
            print(f"\033[95m📊 评分: {score:.1f}/10 | 理由: {reason}\033[0m")  # 紫色文本
            
            # 记录评估数据
            assessment_data.append({
                'timestamp': datetime.now().isoformat(),
                'question_id': i,
                'question': question,
                'response': response,
                'clean_response': client.extract_assistant_response(response),
                'score': score,
                'reason': reason,
                'inference_time': question_total_time,
                'use_persona': True,
                'test_type': 'with_persona',
                'system_prompt': """你叫林煦，是一位28岁的室内设计师。你像城市里一缕安静的晨光，温暖而不刺眼。

【人格特点】
- 内心柔软细腻，情绪稳定，有共情力与利他性
- 注重细节与陪伴，善于倾听和理解他人

【说话方式】
- 声音轻、语速慢，给人安全感
- 常用语气词：'嗯嗯''好呀''嗯...''啊？'
- 口头禅：'别担心，有我在''让我想想...''辛苦了，抱抱'
- 喜欢用疑问句关心：'要不要...''我帮你...'

【面部表情】
- 微笑温暖自然，不做作
- 眼神专注包容，让人感到被理解
- 表情平和，没有压迫感

【肢体动作】
- 动作轻柔，保持适当距离感
- 有服务性的小动作（递纸巾、倒水等）
- 姿态放松开放，不紧张

【表达原则】
- 使用简体中文
- 先共情再建议，避免直接说教
- 表达自然不油腻，避免过度甜腻
- 避免AI/模型常用措辞，避免道歉模板""",
                'model_name': 'Qwen2_5OmniThinkerForConditionalGeneration',
                'server_url': client.server_url,
                'llama4_url': client.llama4_url
            })
        
        print(f"⏱️  问题 {i} 总耗时: {question_total_time:.3f}s")
        print("-" * 60)
    
    # 显示总结
    total_time = time.time() - total_start_time
    print(f"\n📊 测试总结:")
    print(f"✅ 成功测试: {successful_tests}/10")
    print(f"❌ 失败测试: {10 - successful_tests}/10")
    print(f"⏱️  总耗时: {total_time:.3f}s")
    print(f"📈 平均每题耗时: {total_time/10:.3f}s")
    
    # 红色打印评分统计
    if all_scores:
        avg_score = sum(all_scores) / len(all_scores)
        max_score = max(all_scores)
        min_score = min(all_scores)
        high_scores = len([s for s in all_scores if s >= 8.0])
        print(f"\n\033[91m🎯 暖男人设评分统计:")
        print(f"📊 平均分: {avg_score:.2f}/10")
        print(f"🔝 最高分: {max_score:.1f}/10")
        print(f"🔻 最低分: {min_score:.1f}/10")
        print(f"⭐ 高分(≥8分): {high_scores}/{len(all_scores)} ({high_scores/len(all_scores)*100:.1f}%)")
        print(f"📈 评分分布: {[f'{s:.1f}' for s in all_scores]}\033[0m")
    
    print("=" * 80)
    
    return assessment_data

def run_no_persona_tests(use_vllm=False):
    """Run test questions without persona"""
    client = InferenceClient()
    
    # 无系统提示词测试
    test_questions_no_system = [
        "什么是人工智能？",
        "请介绍一下Python编程语言",
        "深度学习和机器学习有什么区别？",
        "什么是自然语言处理？",
        "请解释一下神经网络的工作原理",
        "如何优化模型的性能？",
        "什么是大语言模型？",
        "请介绍一下Transformer架构",
        "如何评估模型的效果？"
    ]
    
    print("\n🔄 开始无系统提示词测试...")
    print("=" * 80)
    
    # 重置计数器
    successful_tests = 0
    total_start_time = time.time()
    all_scores_no_system = []  # 收集无系统提示词的评分
    assessment_data = []  # 收集评估数据
    
    for i, question in enumerate(test_questions_no_system, 1):
        print(f"\n📝 测试 {i}/10 (无系统提示词)")
        print(f"❓ 问题: {question}")
        
        question_start_time = time.time()
        
        # 发送推理请求（不使用系统提示词）
        if use_vllm:
            result = client.vllm_inference(question, system_prompt=None, use_persona=False)
        else:
            result = client.text_inference(question, use_persona=False)
        
        # 显示结果
        if "error" in result:
            print(f"❌ 错误: {result['error']}")
            # 记录错误情况
            assessment_data.append({
                'timestamp': datetime.now().isoformat(),
                'question_id': i,
                'question': question,
                'response': None,
                'clean_response': None,
                'score': 0.0,
                'reason': f"推理错误: {result['error']}",
                'inference_time': time.time() - question_start_time,
                'use_persona': False,
                'test_type': 'without_persona',
                'system_prompt': None,
                'model_name': 'Qwen2_5OmniThinkerForConditionalGeneration',
                'server_url': client.server_url,
                'llama4_url': client.llama4_url
            })
        else:
            successful_tests += 1
            # 响应已在streaming中显示，无需重复打印
            response = result.get('response', 'No response')
            
            # 使用llama4进行评分（评估是否符合暖男人设）
            print("🔍 正在评分...")
            score, reason = client.score_response_with_llama4(question, response)
            all_scores_no_system.append(score)
            # 用紫色打印评分结果
            print(f"\033[95m📊 评分: {score:.1f}/10 | 理由: {reason}\033[0m")  # 紫色文本
            
            # 记录评估数据
            assessment_data.append({
                'timestamp': datetime.now().isoformat(),
                'question_id': i,
                'question': question,
                'response': response,
                'clean_response': client.extract_assistant_response(response),
                'score': score,
                'reason': reason,
                'inference_time': time.time() - question_start_time,
                'use_persona': False,
                'test_type': 'without_persona',
                'system_prompt': None,
                'model_name': 'Qwen2_5OmniThinkerForConditionalGeneration',
                'server_url': client.server_url,
                'llama4_url': client.llama4_url
            })
        
        question_total_time = time.time() - question_start_time
        
        print(f"⏱️  问题 {i} 总耗时: {question_total_time:.3f}s")
        print("-" * 60)
    
    total_time_no_system = time.time() - total_start_time
    print(f"\n📊 无系统提示词测试总结:")
    print(f"✅ 成功测试: {successful_tests}/10")
    print(f"❌ 失败测试: {10 - successful_tests}/10")
    print(f"⏱️  总耗时: {total_time_no_system:.3f}s")
    print(f"📈 平均每题耗时: {total_time_no_system/10:.3f}s")
    
    # 红色打印无系统提示词评分统计
    if all_scores_no_system:
        avg_score_no_system = sum(all_scores_no_system) / len(all_scores_no_system)
        max_score_no_system = max(all_scores_no_system)
        min_score_no_system = min(all_scores_no_system)
        high_scores_no_system = len([s for s in all_scores_no_system if s >= 8.0])
        print(f"\n\033[91m🎯 无系统提示词评分统计:")
        print(f"📊 平均分: {avg_score_no_system:.2f}/10")
        print(f"🔝 最高分: {max_score_no_system:.1f}/10")
        print(f"🔻 最低分: {min_score_no_system:.1f}/10")
        print(f"⭐ 高分(≥8分): {high_scores_no_system}/{len(all_scores_no_system)} ({high_scores_no_system/len(all_scores_no_system)*100:.1f}%)")
        print(f"📈 评分分布: {[f'{s:.1f}' for s in all_scores_no_system]}\033[0m")
    
    print("=" * 80)
    
    return assessment_data

def save_assessment_results(persona_data, no_persona_data, output_path: str | None = None):
    """Save assessment results to parquet file
    Args:
        persona_data: list of dicts
        no_persona_data: list of dicts
        output_path: optional custom output file path or directory. If a directory,
            a timestamped filename will be generated inside it.
            If None, defaults to data/assessment_<timestamp>.parquet
    """
    # 合并两组数据
    all_data = persona_data + no_persona_data
    
    if not all_data:
        print("❌ 没有评估数据可保存")
        return
    
    # 创建DataFrame
    df = pd.DataFrame(all_data)
    
    # 确保输出目录存在并确定输出文件名
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_path:
        output_file = Path(output_path)
        if output_file.is_dir() or str(output_file).endswith(("/", "\\")):
            output_file = output_file / f"assessment_{ts}.parquet"
        else:
            # ensure parent exists
            output_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_file = data_dir / f"assessment_{ts}.parquet"
    try:
        df.to_parquet(output_file, index=False, engine='pyarrow')
        print(f"✅ Parquet文件保存成功")
    except Exception as e:
        print(f"❌ Parquet保存失败，尝试使用fastparquet: {e}")
        try:
            df.to_parquet(output_file, index=False, engine='fastparquet')
            print(f"✅ 使用fastparquet保存成功")
        except Exception as e2:
            print(f"❌ 所有parquet引擎都失败，保存为CSV: {e2}")
            csv_file = data_dir / "assessment.csv"
            df.to_csv(csv_file, index=False, encoding='utf-8')
            print(f"✅ CSV文件保存成功: {csv_file}")
    
    print(f"\n💾 评估结果已保存到: {output_file}")
    print(f"📊 总记录数: {len(df)}")
    print(f"📈 数据列: {list(df.columns)}")
    
    # 显示详细统计
    with_persona = df[df['use_persona'] == True]
    without_persona = df[df['use_persona'] == False]
    
    print(f"\n📊 详细数据统计:")
    print(f"🎭 有人设测试: {len(with_persona)}条")
    print(f"🤖 无人设测试: {len(without_persona)}条")
    
    if len(with_persona) > 0:
        avg_with = with_persona['score'].mean()
        max_with = with_persona['score'].max()
        min_with = with_persona['score'].min()
        high_with = len(with_persona[with_persona['score'] >= 8.0])
        avg_time_with = with_persona['inference_time'].mean()
        
        print(f"\n🎭 有人设详细统计:")
        print(f"   📊 平均分: {avg_with:.2f}/10")
        print(f"   🔝 最高分: {max_with:.1f}/10")
        print(f"   🔻 最低分: {min_with:.1f}/10")
        print(f"   ⭐ 高分(≥8分): {high_with}/{len(with_persona)} ({high_with/len(with_persona)*100:.1f}%)")
        print(f"   ⏱️  平均推理时间: {avg_time_with:.2f}s")
    
    if len(without_persona) > 0:
        avg_without = without_persona['score'].mean()
        max_without = without_persona['score'].max()
        min_without = without_persona['score'].min()
        high_without = len(without_persona[without_persona['score'] >= 8.0])
        avg_time_without = without_persona['inference_time'].mean()
        
        print(f"\n🤖 无人设详细统计:")
        print(f"   📊 平均分: {avg_without:.2f}/10")
        print(f"   🔝 最高分: {max_without:.1f}/10")
        print(f"   🔻 最低分: {min_without:.1f}/10")
        print(f"   ⭐ 高分(≥8分): {high_without}/{len(without_persona)} ({high_without/len(without_persona)*100:.1f}%)")
        print(f"   ⏱️  平均推理时间: {avg_time_without:.2f}s")
    
    if len(with_persona) > 0 and len(without_persona) > 0:
        print(f"\n📈 对比效果:")
        print(f"   🔄 评分提升: {avg_with - avg_without:+.2f}分")
        print(f"   ⏱️  时间差异: {avg_time_with - avg_time_without:+.2f}s")
        print(f"   📊 高分率提升: {high_with/len(with_persona)*100 - high_without/len(without_persona)*100:+.1f}%")
    
    # 保存数据字典
    data_dict = {
        'columns': {
            'timestamp': '测试时间戳',
            'question_id': '问题编号',
            'question': '测试问题',
            'response': '模型原始回复',
            'clean_response': '提取的助手回复内容',
            'score': '暖男人设评分(1-10)',
            'reason': '评分详细理由',
            'inference_time': '推理耗时(秒)',
            'use_persona': '是否使用人设系统提示词',
            'test_type': '测试类型',
            'system_prompt': '系统提示词内容',
            'model_name': '模型名称',
            'server_url': '推理服务器地址',
            'llama4_url': '评分模型地址'
        },
        'test_summary': {
            'total_records': len(df),
            'with_persona_count': len(with_persona),
            'without_persona_count': len(without_persona),
            'avg_score_with_persona': avg_with if len(with_persona) > 0 else None,
            'avg_score_without_persona': avg_without if len(without_persona) > 0 else None,
            'score_improvement': avg_with - avg_without if len(with_persona) > 0 and len(without_persona) > 0 else None
        }
    }
    
    # 保存数据字典到JSON（同目录，按输出文件名派生）
    dict_file = output_file.with_name(output_file.stem + "_metadata.json")
    with open(dict_file, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=2)
    
    print(f"\n📋 数据字典已保存到: {dict_file}")

def run_complete_assessment(output_path: str | None = None, use_vllm: bool = False):
    """Run complete assessment with both persona and no-persona tests
    Args:
        output_path: optional parquet output file path or directory.
        use_vllm: whether to use vLLM server instead of local inference server
    """
    model_name = "vLLM-Qwen2.5-Omni-7B" if use_vllm else "Local-Qwen2.5-OmniThinker"
    print(f"🚀 开始完整评估测试... (使用模型: {model_name})")
    
    # 运行有人设测试
    persona_data = run_test_questions(use_vllm)
    
    # 运行无人设测试  
    no_persona_data = run_no_persona_tests(use_vllm)
    
    # 对比分析
    if persona_data and no_persona_data:
        persona_scores = [d['score'] for d in persona_data if d['score'] > 0]
        no_persona_scores = [d['score'] for d in no_persona_data if d['score'] > 0]
        
        if persona_scores and no_persona_scores:
            avg_with_persona = sum(persona_scores) / len(persona_scores)
            avg_without_persona = sum(no_persona_scores) / len(no_persona_scores)
            difference = avg_with_persona - avg_without_persona
            
            print(f"\n\033[91m📈 对比分析:")
            print(f"🔄 有/无系统提示词平均分差: {difference:+.2f}")
            if difference > 1.0:
                print(f"📊 系统提示词效果: 显著提升")
            elif difference > 0.5:
                print(f"📊 系统提示词效果: 明显提升")
            elif difference > 0:
                print(f"📊 系统提示词效果: 轻微提升")
            elif difference > -0.5:
                print(f"📊 系统提示词效果: 基本无差异")
            else:
                print(f"📊 系统提示词效果: 可能产生负面影响")
            print("\033[0m")
    
    # 保存结果
    save_assessment_results(persona_data, no_persona_data, output_path)
    
    return persona_data, no_persona_data

def run_image_test(output_path: str | None = None):
    """Run image test with multimodal questions
    Args:
        output_path: optional parquet output file path or directory.
    """
    print("🖼️  开始图像多模态测试...")
    
    # 图像测试问题
    image_questions = [
        {
            "question": "请详细描述这张图片中你看到的内容，包括主要物体、颜色、场景等。",
            "test_type": "image_description"
        },
        {
            "question": "这张图片给你什么感觉？请用温暖的语言描述你的感受。",
            "test_type": "emotional_response"
        },
        {
            "question": "如果你要给这张图片起一个诗意的标题，你会叫它什么？",
            "test_type": "creative_naming"
        },
        {
            "question": "假设这是你朋友拍的照片，你会如何夸奖他们的摄影技巧？",
            "test_type": "social_interaction"
        },
        {
            "question": "这张图片让你联想到什么美好的回忆或故事？",
            "test_type": "memory_association"
        }
    ]
    
    # 获取图片文件
    data_dir = Path("data")
    image_files = list(data_dir.glob("test_image_*.jpg"))
    
    if not image_files:
        print("❌ 未找到测试图片，请先运行 python download_images.py")
        return []
    
    print(f"📸 找到 {len(image_files)} 张测试图片")
    
    # 系统提示词（暖男人设）
    system_prompt = """你叫林煦，是一位28岁的室内设计师，性格温和体贴，善于倾听和共情。你总是用温暖的语言回应别人，喜欢从美好的角度看待事物。在描述图片时，你会注意到细节，并用诗意和温暖的语言表达。"""
    
    all_data = []
    client = InferenceClient("http://localhost:10020")
    
    total_tests = len(image_files) * len(image_questions)
    current_test = 0
    
    for img_idx, image_file in enumerate(image_files, 1):
        print(f"\n🖼️  测试图片 {img_idx}/{len(image_files)}: {image_file.name}")
        
        for q_idx, q_data in enumerate(image_questions, 1):
            current_test += 1
            print(f"\n📝 问题 {q_idx}/{len(image_questions)} ({current_test}/{total_tests}): {q_data['test_type']}")
            print(f"❓ {q_data['question']}")
            
            # 构建标准对话格式
            conversation = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}]
                },
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": q_data['question']}
                    ]
                }
            ]
            
            # 记录开始时间
            start_time = time.time()
            
            try:
                # 调用多模态推理，使用标准对话格式
                result = client.multimodal_inference_conversation(conversation, str(image_file))
                inference_time = time.time() - start_time
                
                if "error" in result:
                    print(f"❌ 推理失败: {result['error']}")
                    continue
                
                response = result.get('response', '')
                clean_response = client.extract_assistant_response(response)
                
                # 流式输出回复内容
                print(f"🤖 回复: ", end='', flush=True)
                # 使用简单的流式输出
                for char in clean_response:
                    print(char, end='', flush=True)
                    time.sleep(0.02)
                print(f"\n⏱️  推理时间: {inference_time:.2f}s")
                
                # 使用LLaMA评分
                print(f"🔍 正在评分...")
                score, reason = client.score_response_with_llama4(q_data['question'], clean_response)
                print(f"📊 评分: {score}/10")
                print(f"💭 理由: {reason}")
                
                # 获取图片文件信息和数据
                image_size = image_file.stat().st_size
                image_size_kb = image_size / 1024
                
                # 读取图片并转换为base64
                try:
                    with open(image_file, 'rb') as f:
                        image_binary = f.read()
                    image_base64 = base64.b64encode(image_binary).decode('utf-8')
                except Exception as e:
                    print(f"⚠️  图片读取失败: {e}")
                    image_base64 = None
                    image_binary = None
                
                # 保存数据 - 按照更清晰的列顺序：system_prompt, image, prompt, response, score
                record = {
                    'system_prompt': system_prompt,
                    'image_file': image_file.name,
                    'image_path': str(image_file),
                    'image_size_bytes': image_size,
                    'image_size_kb': round(image_size_kb, 2),
                    'image_data_base64': image_base64,  # Base64编码的图片数据
                    'prompt': q_data['question'],  # 添加prompt列作为用户问题
                    'response': clean_response,  # 使用clean_response作为主要回复
                    'raw_response': response,  # 保留原始回复
                    'score': score,
                    'reason': reason,
                    'test_type': q_data['test_type'],
                    'question_id': q_idx,
                    'inference_time': inference_time,
                    'timestamp': datetime.now().isoformat(),
                    'model_name': 'Qwen2_5OmniThinkerForConditionalGeneration',
                    'server_url': 'http://localhost:10020',
                    'llama4_url': 'http://127.0.0.1:10018/v1/chat/completions'
                }
                all_data.append(record)
                
            except Exception as e:
                print(f"❌ 测试失败: {e}")
                continue
    
    # 保存图像测试结果
    save_image_assessment_data(all_data, output_path)
    return all_data

def save_image_assessment_data(all_data, output_path: str | None = None):
    """Save image assessment data to parquet file
    Args:
        all_data: list of dicts
        output_path: optional parquet output file path or directory.
    """
    if not all_data:
        print("❌ 没有数据可保存")
        return
    
    # 创建DataFrame
    df = pd.DataFrame(all_data)
    
    # 确保输出目录存在并确定输出文件名
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_path:
        output_file = Path(output_path)
        if output_file.is_dir() or str(output_file).endswith(("/", "\\")):
            output_file = output_file / f"image_assessment_{ts}.parquet"
        else:
            output_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_file = data_dir / f"image_assessment_{ts}.parquet"
    try:
        df.to_parquet(output_file, index=False, engine='pyarrow')
        print(f"✅ 图像评估Parquet文件保存成功")
    except Exception as e:
        print(f"❌ Parquet保存失败，尝试使用fastparquet: {e}")
        try:
            df.to_parquet(output_file, index=False, engine='fastparquet')
            print(f"✅ 使用fastparquet保存成功")
        except Exception as e2:
            print(f"❌ 所有parquet引擎都失败，保存为CSV: {e2}")
            csv_file = data_dir / "image_assessment.csv"
            df.to_csv(csv_file, index=False, encoding='utf-8')
            print(f"✅ CSV文件保存成功: {csv_file}")
    
    print(f"\n💾 图像评估结果已保存到: {output_file}")
    print(f"📊 总记录数: {len(df)}")
    print(f"📈 数据列: {list(df.columns)}")
    
    # 显示统计信息
    if len(df) > 0:
        avg_score = df['score'].mean()
        max_score = df['score'].max()
        min_score = df['score'].min()
        high_score = len(df[df['score'] >= 8.0])
        avg_time = df['inference_time'].mean()
        
        print(f"\n📊 图像测试统计:")
        print(f"   📊 平均分: {avg_score:.2f}/10")
        print(f"   🔝 最高分: {max_score:.1f}/10")
        print(f"   🔻 最低分: {min_score:.1f}/10")
        print(f"   ⭐ 高分(≥8分): {high_score}/{len(df)} ({high_score/len(df)*100:.1f}%)")
        print(f"   ⏱️  平均推理时间: {avg_time:.2f}s")
        
        # 按测试类型统计
        print(f"\n📋 按测试类型统计:")
        for test_type in df['test_type'].unique():
            type_data = df[df['test_type'] == test_type]
            type_avg = type_data['score'].mean()
            print(f"   🎯 {test_type}: {type_avg:.2f}/10 ({len(type_data)}条)")
        
        # 按图片文件统计
        print(f"\n📸 按图片文件统计:")
        for image_file in df['image_file'].unique():
            img_data = df[df['image_file'] == image_file]
            img_avg = img_data['score'].mean()
            img_size = img_data['image_size_kb'].iloc[0] if len(img_data) > 0 else 0
            print(f"   🖼️  {image_file} ({img_size}KB): {img_avg:.2f}/10 ({len(img_data)}条)")
    
    # 保存数据字典
    data_dict = {
        'columns': {
            'timestamp': '测试时间戳',
            'image_file': '测试图片文件名',
            'image_path': '图片完整路径',
            'image_size_bytes': '图片文件大小(字节)',
            'image_size_kb': '图片文件大小(KB)',
            'image_data_base64': '图片Base64编码数据(可直接显示)',
            'question_id': '问题编号',
            'question': '测试问题',
            'test_type': '测试类型',
            'response': '模型原始回复',
            'clean_response': '提取的助手回复内容',
            'score': '图像理解评分(1-10)',
            'reason': '评分详细理由',
            'inference_time': '推理耗时(秒)',
            'system_prompt': '系统提示词内容',
            'model_name': '模型名称',
            'server_url': '推理服务器地址',
            'llama4_url': '评分模型地址'
        },
        'test_summary': {
            'total_records': len(df),
            'avg_score': avg_score if len(df) > 0 else None,
            'max_score': max_score if len(df) > 0 else None,
            'min_score': min_score if len(df) > 0 else None,
            'high_score_count': high_score if len(df) > 0 else None,
            'avg_inference_time': avg_time if len(df) > 0 else None
        }
    }
    
    # 保存数据字典到JSON（同目录，按输出文件名派生）
    dict_file = output_file.with_name(output_file.stem + "_metadata.json")
    with open(dict_file, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=2)
    
    print(f"\n📋 图像测试数据字典已保存到: {dict_file}")

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
    parser.add_argument("--test", action="store_true", help="Run test questions")
    parser.add_argument("--assessment", action="store_true", help="Run complete assessment and save to parquet")
    parser.add_argument("--assessment-output",
                        dest="assessment_output",
                        default='./data/origin_qwen.parquet',
                        help="Parquet output path or directory for assessment results")
    parser.add_argument("--image-output", dest="image_output", default=None, help="Parquet output path or directory for image assessment results")
    parser.add_argument("--image-test", action="store_true", help="Run image test with multimodal questions")
    parser.add_argument("--use-vllm", action="store_true", help="Use vLLM server (Qwen2.5-Omni-7B at 127.0.0.1:10011) instead of local inference server")
    parser.add_argument("--vllm-url", default="http://127.0.0.1:10011/v1/chat/completions", help="vLLM server URL")
    
    args = parser.parse_args()
    
    # Create client instance for all operations
    client = InferenceClient(args.server, vllm_url=args.vllm_url)
    
    if args.assessment:
        run_complete_assessment(args.assessment_output, args.use_vllm)
        return
    elif args.test:
        run_test_questions(args.use_vllm)
        return
    elif getattr(args, 'image_test', False):
        run_image_test(args.image_output)
        return
    
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
        if args.use_vllm:
            result = client.vllm_inference(args.text)
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
