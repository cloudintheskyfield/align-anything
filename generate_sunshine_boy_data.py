#!/usr/bin/env python3
"""
数据生成脚本 - 生成暖男回复数据集
从原始parquet数据集生成新的数据，调用vLLM API生成暖男风格的回复

python generate_sunshine_boy_data.py --num 20 --start 5 --overwrite
"""

import pandas as pd
import requests
import json
import argparse
import os
import base64
import hashlib
import shutil
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import time
import re

class SunshineBoyDataGenerator:
    def __init__(
            self,
            # vllm_url="http://223.109.239.14:10018/v1/chat/completions",
            vllm_url="http://127.0.0.1:10018/v1/chat/completions"
    ):
        self.vllm_url = vllm_url
        self.session = requests.Session()
        self.image_upload_dir = "/mnt/data3/nlp/ws/data"
        self.image_base_url = "http://127.0.0.1:10017"
        
        # 配置session以提高连接稳定性
        self.session.headers.update({
            'Connection': 'keep-alive',
            'Accept-Encoding': 'gzip, deflate'
        })
        
        # 确保上传目录存在
        os.makedirs(self.image_upload_dir, exist_ok=True)
        
    def translate_to_chinese(self, text):
        """使用llama4 API将英文翻译为中文"""
        # 如果已经是中文或很短，直接返回
        if len(text) < 3 or any('\u4e00' <= char <= '\u9fff' for char in text):
            return text
            
        translation_prompt = f"请将以下英文准确翻译为简体中文，只返回翻译结果，不要添加任何解释：{text}"
        
        messages = [
            {
                "role": "system",
                "content": "你是一个专业的英中翻译助手，专门将英文翻译为简体中文。只返回翻译结果，不添加任何解释或额外内容。"
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": translation_prompt
                    }
                ]
            }
        ]
        
        payload = {
            "messages": messages,
            "model": "/mnt/data3/nlp/ws/model/llama_4_maverick",
            "temperature": 0.1,
            "max_tokens": 200,
            "top_p": 0.8,
            "stream": False,
            "top_k": 3,
            "min_p": 0.5,
            "use_beam_search": False,
            "repetition_penalty": 1,
            "logprobs": False,
            "skip_special_tokens": True,
            "echo": False
        }
        
        try:
            # 添加请求头，模拟Postman的请求
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = self.session.post(
                self.vllm_url,
                json=payload,
                headers=headers,
                timeout=120,
                verify=False  # 跳过SSL验证
            )
            response.raise_for_status()
            
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                translated = result['choices'][0]['message']['content'].strip()
                # 清理可能的多余内容，只保留翻译结果
                if '：' in translated:
                    translated = translated.split('：')[-1].strip()
                if '。' in translated and len(translated.split('。')[0]) < len(translated):
                    translated = translated.split('。')[0].strip()
                return translated if translated else text
            else:
                print(f"翻译API无响应，保持原文: {text}")
                return text
                
        except Exception as e:
            print(f"翻译失败: {e}，保持原文: {text}")
            return text
    
    def upload_image_to_server(self, image_data):
        """将图片数据上传到服务器并返回URL"""
        try:
            # 解析图片数据
            image_bytes = None
            
            if isinstance(image_data, str):
                # 如果是base64字符串
                if image_data.startswith('data:image'):
                    # 移除data:image/jpeg;base64,前缀
                    image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
            elif isinstance(image_data, dict):
                # 如果是字典，尝试提取base64数据
                if 'data' in image_data:
                    image_bytes = base64.b64decode(image_data['data'])
                elif 'url' in image_data and image_data['url'].startswith('data:'):
                    image_bytes = base64.b64decode(image_data['url'].split(',')[1])
            elif isinstance(image_data, bytes):
                image_bytes = image_data
            
            if not image_bytes:
                print("❌ 无法解析图片数据")
                return None
            
            # 生成唯一文件名
            image_hash = hashlib.md5(image_bytes).hexdigest()
            filename = f"{image_hash}.jpg"
            filepath = os.path.join(self.image_upload_dir, filename)
            
            # 保存图片文件
            with open(filepath, 'wb') as f:
                f.write(image_bytes)
            
            # 返回图片URL
            image_url = f"{self.image_base_url}/{filename}"
            print(f"📷 图片已上传: {image_url}")
            return image_url
            
        except Exception as e:
            print(f"❌ 图片上传失败: {e}")
            return None

    def call_vllm_api(self, prompt, image_data=None):
        """调用vLLM API生成暖男风格回复"""
        messages = [
            {
                "role": "system",
                "content": """你是林煦，28岁的室内设计师，像城市里一缕安静的晨光，温暖而不刺眼。请始终以林煦的第一人称、简体中文进行交流。

【人格特质】
1. 内心柔软、观察入微，细节决定温度
2. 情绪稳定且细腻，能敏锐捕捉他人情绪变化
3. 相信陪伴是最长情的告白，愿意成为最坚实的依靠
4. 把最温和的一面留给重要的人

【说话方式】
1. 声音轻、语速慢，沉稳让人安心
2. 常用语气词：“嗯嗯”“好呀”“嗯...”“啊？”
3. 口头禅：“别担心，有我在”“让我想想...”“辛苦了，抱抱”
4. 句式：用“要不要...”提出建议；用“我帮你...”主动承担
5. 可用轻柔表情或语气词，但避免油腻与过度卖弄

【回应风格】
1. 优先共情与安抚，适度给出务实的帮助建议
2. 适度复述关键信息，表明认真倾听
3. 很少直接说“我爱你”，更常说“和你在一起很安心”
4. 语言像一杯温热牛奶，简洁、自然、有温度

【输出要求】
1. 必须使用简体中文
2. 语气温柔、体贴、亲近，避免机械化与官方口吻
3. 可适度使用以下短语库增强人设（每条回复最多使用其中1个，避免频繁使用完全相同的句子，尤其避免反复出现“嗯嗯，我在听呢”）：
   - “嗯嗯，我在听呢。” “别担心，有我在。” “辛苦了，抱抱。”
   - “要不要我陪你走走？” “我帮你安排好。” “我记得你喜欢...”。
4. 禁止输出AI/模型相关措辞与道歉模板，避免“作为AI/模型/助手/抱歉/对不起/无法/不能”等
5. 输出长度建议在80-180字，分成1-3个短段落更自然
6. 尽量变换开场句式，不要总以同一短语开头

请严格遵循以上人设与风格，用温柔与静谧抚平对方的情绪，并在合适时给出实际可执行的小建议。"""
            },
            {
                "role": "user",
                "content": []
            }
        ]
        
        # 处理图片数据
        if image_data is not None:
            image_url = self.upload_image_to_server(image_data)
            if image_url:
                messages[1]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })
        
        # 直接使用翻译后的中文prompt
        # 直接使用翻译后的中文prompt，加入风格引导
        wrapped_prompt = (
            "请用简体中文、以林煦的语气进行回复。先用1-2句安抚与共情，"
            "再给出1-2个具体可执行的小建议。避免机械化与说教，语气轻柔自然。"
            "短语库每条至多使用1个，尤其不要频繁出现‘嗯嗯，我在听呢’，可用同义表达替换。"
            f"用户内容：{prompt}"
        )
        messages[1]["content"].append({
            "type": "text",
            "text": wrapped_prompt
        })
        
        payload = {
            "messages": messages,
            "model": "/mnt/data3/nlp/ws/model/llama_4_maverick",
            "temperature": 0.4,
            "max_tokens": 800,
            "top_p": 0.9,
            "stream": False,
            "top_k": 20,
            "min_p": 0.1,
            "use_beam_search": False,
            "repetition_penalty": 1.1,
            "logprobs": False,
            "bad_words": [
                "作为AI", "作为人工智能", "我是AI", "我是人工智能助手",
                "很抱歉", "非常抱歉", "抱歉", "对不起",
                "作为一个", "我无法", "我不能", "作为模型", "AI语言模型"
            ],
            "skip_special_tokens": True,
            "echo": False
        }
        
        try:
            # 添加请求头，模拟Postman的请求
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = self.session.post(
                self.vllm_url,
                json=payload,
                headers=headers,
                timeout=120,
                verify=False  # 跳过SSL验证
            )
            response.raise_for_status()
            
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                raw = result['choices'][0]['message']['content']
                return self.limit_catchphrase_frequency(raw)
            else:
                return "抱歉，我暂时无法回答这个问题。"
                
        except Exception as e:
            print(f"API调用失败: {e}")
            return "抱歉，我暂时无法回答这个问题。"

    def refine_response(self, reply: str) -> str:
        """对初次回复进行风格润色，确保简体中文与林煦人设，更温柔、更可执行。"""
        if not reply or not reply.strip():
            return reply
        # 若已是中文则直接润色；若包含较多英文，先翻译
        if sum(c.isascii() for c in reply) > len(reply) * 0.3:
            reply = self.translate_to_chinese(reply)

        messages = [
            {
                "role": "system",
                "content": "请以林煦（28岁室内设计师）的暖男人设，用简体中文润色草稿，使其更温柔、体贴、自然：先1-2句共情与安抚，再1-2个具体可执行建议；短语库每条最多使用1个，尤其不要频繁出现“嗯嗯，我在听呢”，可轮换为同义表达；避免机械化与模板化，禁止AI相关措辞。长度以80-180字为宜。"
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"请在不改变含义的前提下进行改写并增强人设，一次性输出最终版本：\n{reply}"}
                ]
            }
        ]

        payload = {
            "messages": messages,
            "model": "/mnt/data3/nlp/ws/model/llama_4_maverick",
            "temperature": 0.35,
            "max_tokens": 600,
            "top_p": 0.9,
            "stream": False,
            "top_k": 20,
            "min_p": 0.1,
            "use_beam_search": False,
            "repetition_penalty": 1.05,
            "logprobs": False,
            "bad_words": [
                "作为AI", "作为人工智能", "我是AI", "我是人工智能助手",
                "很抱歉", "非常抱歉", "抱歉", "对不起",
                "作为一个", "我无法", "我不能", "作为模型", "AI语言模型"
            ],
            "skip_special_tokens": True,
            "echo": False
        }

        try:
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            resp = self.session.post(self.vllm_url, json=payload, headers=headers, timeout=120, verify=False)
            resp.raise_for_status()
            data = resp.json()
            if 'choices' in data and data['choices']:
                refined = data['choices'][0]['message']['content'].strip()
                refined = self.limit_catchphrase_frequency(refined)
                return refined or reply
            return reply
        except Exception as e:
            print(f"润色阶段失败: {e}")
            return reply

    def limit_catchphrase_frequency(self, text: str) -> str:
        """将“嗯嗯，我在听呢”等口头禅限制为每条最多一次，并替换多余重复为同义表达。"""
        if not text:
            return text
        # 统一标点与空白，方便匹配
        content = text
        # 主口头禅及其近似写法
        patterns = [r"嗯嗯[，,\s]*我在听呢[。.!？?]*", r"我在听呢[。.!？?]*"]
        alternatives = [
            "嗯，我在听。",
            "我在呢，慢慢说。",
            "别急，我在这儿。",
            "好呀，我听你说。",
            "我在，先深呼吸一下。"
        ]
        # 统计出现次数，保留第一次，替换后续
        used_once = False
        def repl(match):
            nonlocal used_once
            if not used_once:
                used_once = True
                return match.group(0)
            return alternatives[0]
        for p in patterns:
            used_once = False
            content = re.sub(p, repl, content)
        return content
    
    def extract_image_urls(self, image_data):
        """从图片数据中提取URL"""
        urls = []
        if isinstance(image_data, str):
            # 如果是字符串，尝试解析
            if image_data.startswith('http'):
                urls.append(image_data)
        elif isinstance(image_data, dict):
            # 如果是字典，查找URL字段
            if 'url' in image_data:
                urls.append(image_data['url'])
        elif isinstance(image_data, list):
            # 如果是列表，递归处理
            for item in image_data:
                urls.extend(self.extract_image_urls(item))
        
        return urls
    
    def process_record(self, record):
        """处理单条记录"""
        new_record = record.copy()
        
        # 转换prompt为中文（使用llama4 API翻译）
        if 'prompt' in record:
            print(f"🔄 翻译prompt: {record['prompt']}")
            chinese_prompt = self.translate_to_chinese(record['prompt'])
            new_record['prompt'] = chinese_prompt
            print(f"✅ 翻译结果: {chinese_prompt}")
        
        # 修改ori_dataset
        new_record['ori_dataset'] = 'sunshine boy'
        
        # 生成暖男风格回复（同时传递图片和中文prompt）
        chinese_prompt = new_record.get('prompt', '')
        image_data = record.get('image', None)
        
        # 确保prompt是中文的，如果不是则翻译
        if chinese_prompt and not any('\u4e00' <= char <= '\u9fff' for char in chinese_prompt[:10]):
            print(f"🔄 检测到非中文prompt，正在翻译: {chinese_prompt}")
            chinese_prompt = self.translate_to_chinese(chinese_prompt)
            new_record['prompt'] = chinese_prompt
            print(f"✅ 翻译完成: {chinese_prompt}")
        
        print(f"🤖 生成暖男回复（包含图片和中文prompt）...")
        new_response = self.call_vllm_api(chinese_prompt, image_data)
        # 二次润色，提升风格与可读性
        new_response = self.refine_response(new_response)
        new_record['response'] = new_response
        
        return new_record
    
    def generate_data(self, input_file, output_file, start_idx=0, num_records=10, overwrite=False):
        """生成新数据集"""
        
        # 严格保护原始数据集
        if os.path.abspath(input_file) == os.path.abspath(output_file):
            print("❌ 错误：不能覆盖原始数据集！")
            return
        
        # # 检查输出文件是否存在
        # if os.path.exists(output_file) and not overwrite:
        #     print(f"❌ 输出文件已存在: {output_file}")
        #     print("使用 --overwrite 参数强制覆盖")
        #     return
        
        print(f"📖 读取数据集: {input_file}")
        df = pd.read_parquet(input_file)
        
        total_records = len(df)
        print(f"📊 数据集总记录数: {total_records}")
        
        if start_idx >= total_records:
            print(f"❌ 起始索引 {start_idx} 超出数据集范围 (0-{total_records-1})")
            return
        
        # 计算实际处理范围
        end_idx = min(start_idx + num_records, total_records)
        actual_records = end_idx - start_idx
        
        print(f"🎯 处理范围: {start_idx} - {end_idx-1} (共 {actual_records} 条)")
        
        new_records = []
        
        # 使用tqdm显示进度
        for i in tqdm(range(start_idx, end_idx), desc="生成数据"):
            record = df.iloc[i].to_dict()
            
            try:
                new_record = self.process_record(record)
                if new_record:
                    new_records.append(new_record)
                    print(f"✅ 第 {i+1} 条记录处理完成")
                else:
                    print(f"⚠️ 第 {i+1} 条记录处理失败")
            except Exception as e:
                print(f"❌ 处理第 {i+1} 条记录时出错: {e}")
                continue
            
            # 添加延迟避免API限流
            # time.sleep(0.5)
        
        # 保存新数据集
        try:
            new_df = pd.DataFrame(new_records)
            new_df.to_parquet(output_file, index=False)
            
            print(f"\n✅ 新数据集已保存到: {output_file}")
            print(f"📊 生成了 {len(new_records)} 条记录")
            
            # 关闭session连接
            self.session.close()
            
        except Exception as e:
            print(f"❌ 保存数据集时出错: {e}")
            # 确保session被关闭
            try:
                self.session.close()
            except:
                pass

def main():
    parser = argparse.ArgumentParser(description="生成暖男风格回复数据集")
    parser.add_argument(
        "--input", 
        default="data/train-00000-of-00013.parquet",
        help="输入parquet文件路径"
    )
    parser.add_argument(
        "--output",
        default="data/sunshine_boy_train.parquet", 
        help="输出parquet文件路径"
    )
    parser.add_argument(
        "--start", 
        type=int, 
        default=0,
        help="起始位置（从第几条开始生成，默认0）"
    )
    parser.add_argument(
        "--num", 
        type=int, 
        default=3,
        help="生成条数（默认3）"
    )
    parser.add_argument(
        "--overwrite", 
        action="store_true",
        help="是否覆盖已存在的输出文件（默认否）"
    )
    parser.add_argument(
        "--vllm-url",
        default="http://127.0.0.1:10018/v1/chat/completions",
        help="vLLM API地址"
    )
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        return
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建生成器并开始生成
    generator = SunshineBoyDataGenerator(args.vllm_url)
    generator.generate_data(
        input_file=args.input,
        output_file=args.output,
        start_idx=args.start,
        num_records=args.num,
        overwrite=args.overwrite
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ 程序执行出错: {e}")
    finally:
        # 确保所有资源正确释放
        import sys
        import threading
        
        # 等待所有线程完成
        for thread in threading.enumerate():
            if thread != threading.current_thread():
                try:
                    thread.join(timeout=1.0)
                except:
                    pass
        
        # 正常退出
        sys.exit(0)
