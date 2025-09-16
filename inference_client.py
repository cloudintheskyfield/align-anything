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
import re
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path

class InferenceClient:
    def __init__(self, server_url="http://localhost:10020", llama4_url="http://127.0.0.1:10018/v1/chat/completions"):
        self.server_url = server_url.rstrip('/')
        self.llama4_url = llama4_url
        self.session = requests.Session()
        self.use_streaming = True  # Enable streaming by default
        
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
            "max_tokens": 300,
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

def run_test_questions():
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

def run_no_persona_tests():
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

def save_assessment_results(persona_data, no_persona_data):
    """Save assessment results to parquet file"""
    # 合并两组数据
    all_data = persona_data + no_persona_data
    
    if not all_data:
        print("❌ 没有评估数据可保存")
        return
    
    # 创建DataFrame
    df = pd.DataFrame(all_data)
    
    # 确保data目录存在
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # 保存到parquet文件
    output_file = data_dir / "assessment.parquet"
    df.to_parquet(output_file, index=False)
    
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
    
    # 保存数据字典到JSON
    dict_file = data_dir / "assessment_metadata.json"
    with open(dict_file, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=2)
    
    print(f"\n📋 数据字典已保存到: {dict_file}")

def run_complete_assessment():
    """Run complete assessment with both persona and no-persona tests"""
    print("🚀 开始完整评估测试...")
    
    # 运行有人设测试
    persona_data = run_test_questions()
    
    # 运行无人设测试  
    no_persona_data = run_no_persona_tests()
    
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
    save_assessment_results(persona_data, no_persona_data)
    
    return persona_data, no_persona_data

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
    parser.add_argument("--test", action="store_false", help="Run test questions")
    parser.add_argument("--assessment", action="store_false", help="Run complete assessment and save to parquet")
    
    args = parser.parse_args()
    
    if args.assessment:
        run_complete_assessment()
    elif args.test:
        run_test_questions()
    else:
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
