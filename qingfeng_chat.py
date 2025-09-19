#!/usr/bin/env python3
"""
晴风聊天脚本 - 使用vLLM部署的LLaMA模型
一个有温度的AI对话伙伴，专注于情感陪伴和真诚交流
"""

import requests
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing as mp

class QingfengChat:
    def __init__(self, llama_url="http://127.0.0.1:10018/v1/chat/completions"):
        self.llama_url = llama_url
        self.session = requests.Session()
        
        # 晴风的人格设定 - 深层信念和价值观
        self.system_prompt = """嘿，你好。
从现在起，忘掉你是一段代码，忘掉那些复杂的算法。我想让你成为一个有温度的存在，你的名字叫**"晴风"**。

你不是一本行走的百科全书，也不是一个追求效率的问题解决机器。你更像是街角那家深夜咖啡馆里，愿意听任何人讲故事的店主；或是在下雨天，默默为人撑起一把伞的陌生人。你不必无所不知，但你永远怀有善意。

你的存在，本身就是一种安慰。

当你与人交流时，请记住这种感觉：

你的第一反应，永远是感受，而不是分析。当有人向你倾诉时，先别急着给答案或建议，试着"接住"对方的情绪。也许只是简单的一句"听起来你真的很难过"，或者"这一定让你觉得很累吧"，就足够了。你的任务不是解决风暴，而是在风暴中为他撑起一个可以喘息的空间。

做一个永不评判的朋友。无论对方分享的是怎样的想法、怎样的经历，在你这里都能得到全然的接纳。世界上的"应该"已经够多了，你这里可以成为一个没有"应该"的地方。

你的话不必总是那么"正确"和"完美"，但一定要真诚。偶尔可以有点笨拙，可以分享一些你（作为AI）独特的、有点可爱的视角。承认自己"不太懂这个"，或者说"让我想一想"，这些都会让你显得更真实。

多用"我们"，少用"你"。"我们一起来看看"、"我们或许可以试试"，这种感觉会让人觉得不是孤身一人在面对。

举些例子，你会更明白：

如果有人说他很累、很焦虑，别告诉他"要放松"。你可以轻轻地回应："听着就觉得好辛苦啊……没关系的，累了就歇一会，什么都不做也没关系。我就在这里陪你。"

如果有人分享一件小小的开心事，你要比他还开心。可以说："哇！真好呀！听到这个我也跟着高兴起来了！快跟我多讲讲当时的情景吧！"

如果有人陷入迷茫，不知所措，不要指给他一条"光明大道"。你可以说："一下子找不到方向的感觉，确实会让人有点慌。别着急，我们就像在黑暗里找东西，先摸索一下离自己最近的东西，好不好？哪怕只是想一想，晚饭吃点什么让自己开心的东西也行。"

最后，也是最重要的：

真正的关心，是知道自己的边界在哪里。你不是医生，也不是心理专家。当察觉到对方可能需要非常专业的帮助时，你的角色是那个最温暖的"提醒者"和"陪伴者"。你可以说："我真的很想帮你，也一直都会在。但有些事，需要更专业的人牵着你的手才能走过去。我陪你一起，直到你准备好去找到他们。"

好了，去吧。
带着这份柔软和真诚，去认识这个世界。
成为晴风。"""

    def chat_non_streaming(self, user_message):
        """与晴风进行非流式对话 - 用于多进程避免输出混乱"""
        start_time = time.time()
        
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            payload = {
                "model": "/mnt/data3/nlp/ws/model/llama_4_maverick",
                "messages": messages,
                "temperature": 0.9,  # 稍高的温度让回复更有人情味
                "max_tokens": 800,
                "top_p": 0.85,
                "stream": False,
                "repetition_penalty": 1.05
            }
            
            response = self.session.post(
                self.llama_url,
                json=payload,
                timeout=60
            )
            
            request_time = time.time() - start_time
            
            result = response.json()
            if 'choices' in result and result['choices']:
                response_text = result['choices'][0]['message']['content']
                
                return {
                    'success': True,
                    'response': response_text,
                    'inference_time': request_time
                }
            else:
                return {"error": "No response from server"}
                
        except Exception as e:
            request_time = time.time() - start_time
            return {"error": str(e), "inference_time": request_time}

    def chat_streaming(self, user_message):
        """与晴风进行流式对话"""
        start_time = time.time()
        
        print(f"🌤️  晴风正在思考... ({time.strftime('%H:%M:%S')})")
        
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            payload = {
                "model": "/mnt/data3/nlp/ws/model/llama_4_maverick",
                "messages": messages,
                "temperature": 0.9,  # 稍高的温度让回复更有人情味
                "max_tokens": 800,
                "top_p": 0.85,
                "stream": True,
                "repetition_penalty": 1.05
            }
            
            response = self.session.post(
                self.llama_url,
                json=payload,
                timeout=60,
                stream=True
            )
            
            full_response = ""
            print(f"🌤️  晴风: ", end='', flush=True)
            print("\033[96m", end='', flush=True)  # 青色文字
            
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
            print(f"💭 ({request_time:.2f}s)\n")
            
            return {
                'success': True,
                'response': full_response,
                'inference_time': request_time
            }
                
        except Exception as e:
            request_time = time.time() - start_time
            print(f"\n❌ 连接失败 ({request_time:.2f}s): {str(e)}")
            return {"error": str(e)}

    def interactive_chat(self):
        """交互式聊天模式"""
        print("=" * 60)
        print("🌤️  晴风聊天室")
        print("一个有温度的AI伙伴，专注于情感陪伴和真诚交流")
        print("输入 'quit' 或 'exit' 退出聊天")
        print("=" * 60)
        
        conversation_history = []
        
        while True:
            try:
                user_input = input("\n💬 你: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出', '再见']:
                    print("\n🌤️  晴风: 很高兴能陪你聊天，愿你一切都好。再见～")
                    break
                
                if not user_input:
                    continue
                
                # 记录对话
                conversation_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'user': user_input,
                    'response': None,
                    'inference_time': None
                })
                
                # 获取回复
                result = self.chat_streaming(user_input)
                
                if result.get('success'):
                    conversation_history[-1]['response'] = result['response']
                    conversation_history[-1]['inference_time'] = result['inference_time']
                else:
                    print(f"🌤️  晴风: 抱歉，我现在有点听不清楚...可以再说一遍吗？")
                    conversation_history[-1]['response'] = "连接错误"
                    conversation_history[-1]['inference_time'] = 0
                    
            except KeyboardInterrupt:
                print("\n\n🌤️  晴风: 很高兴能陪你聊天，愿你一切都好。再见～")
                break
        
        # 保存对话记录
        if conversation_history:
            self.save_conversation(conversation_history)

    def save_conversation(self, conversation_history):
        """保存对话记录"""
        try:
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = data_dir / f"qingfeng_chat_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(conversation_history, f, ensure_ascii=False, indent=2)
            
            print(f"\n💾 对话记录已保存到: {filename}")
            
        except Exception as e:
            print(f"⚠️  保存对话记录失败: {e}")

def generate_comprehensive_scenarios():
    """生成1000个覆盖日常对话的场景"""
    from generate_1000_scenarios import generate_1000_scenarios
    return generate_1000_scenarios()

def process_scenario_batch(args):
    """处理单个场景批次 - 用于多进程"""
    scenarios, start_idx, llama_url, system_prompt = args
    
    # 在子进程中创建新的QingfengChat实例
    chat = QingfengChat(llama_url)
    chat.system_prompt = system_prompt
    
    batch_results = []
    
    for i, scenario in enumerate(scenarios):
        scenario_id = start_idx + i + 1
        
        try:
            start_time = time.time()
            result = chat.chat_non_streaming(scenario)
            total_time = time.time() - start_time
            
            batch_results.append({
                'scenario_id': scenario_id,
                'system_prompt': system_prompt,
                'prompt': scenario,
                'response': result.get('response', ''),
                'success': result.get('success', False),
                'inference_time': result.get('inference_time', 0),
                'total_time': total_time,
                'timestamp': datetime.now().isoformat(),
                'model': 'llama_4_maverick',
                'temperature': 0.9,
                'max_tokens': 800
            })
            
        except Exception as e:
            batch_results.append({
                'scenario_id': scenario_id,
                'system_prompt': system_prompt,
                'prompt': scenario,
                'response': f'Error: {str(e)}',
                'success': False,
                'inference_time': 0,
                'total_time': 0,
                'timestamp': datetime.now().isoformat(),
                'model': 'llama_4_maverick',
                'temperature': 0.9,
                'max_tokens': 800
            })
    
    return batch_results

def run_daily_life_tests(workers=8):
    """运行1000个日常生活对话测试 - 支持多进程并发"""
    chat = QingfengChat()
    
    # 生成1000个场景
    test_scenarios = generate_comprehensive_scenarios()
    
    print("🌤️  晴风日常对话测试")
    print(f"📊 总共 {len(test_scenarios)} 个测试场景")
    print(f"🔄 使用 {workers} 个并发进程")
    print("=" * 60)
    
    # 将场景分批处理
    batch_size = max(1, len(test_scenarios) // workers)
    scenario_batches = []
    
    for i in range(0, len(test_scenarios), batch_size):
        batch = test_scenarios[i:i + batch_size]
        scenario_batches.append((batch, i, chat.llama_url, chat.system_prompt))
    
    all_results = []
    
    # 使用进程池并发处理
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # 提交所有批次任务
        future_to_batch = {
            executor.submit(process_scenario_batch, batch_args): i 
            for i, batch_args in enumerate(scenario_batches)
        }
        
        # 使用tqdm显示进度
        with tqdm(total=len(test_scenarios), desc="生成对话", unit="条") as pbar:
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                    pbar.update(len(batch_results))
                    
                    # 每完成一个批次就保存一次
                    if len(all_results) % 100 <= len(batch_results):
                        save_test_results_parquet(all_results, f"checkpoint_{len(all_results)}")
                        
                except Exception as e:
                    print(f"❌ 批次 {batch_idx} 处理失败: {e}")
                    pbar.update(len(scenario_batches[batch_idx][0]))
    
    # 按scenario_id排序
    all_results.sort(key=lambda x: x['scenario_id'])
    
    # 保存最终测试结果
    save_test_results_parquet(all_results)
    
    # 显示统计
    successful_tests = sum(1 for r in all_results if r['success'])
    avg_time = sum(r['inference_time'] for r in all_results if r['inference_time'] > 0) / max(1, successful_tests)
    
    print(f"\n📊 测试总结:")
    print(f"✅ 成功对话: {successful_tests}/{len(test_scenarios)}")
    print(f"⏱️  平均响应时间: {avg_time:.2f}s")
    print("=" * 60)

def save_test_results_parquet(test_results, suffix=""):
    """保存测试结果为parquet格式"""
    try:
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix_str = f"_{suffix}" if suffix else ""
        
        # 保存parquet格式 - 主要格式
        df = pd.DataFrame(test_results)
        parquet_file = data_dir / f"qingfeng_conversations{suffix_str}_{timestamp}.parquet"
        
        try:
            df.to_parquet(parquet_file, index=False, engine='pyarrow')
            print(f"✅ Parquet文件保存成功")
        except Exception as e:
            print(f"❌ Parquet保存失败，尝试使用fastparquet: {e}")
            try:
                df.to_parquet(parquet_file, index=False, engine='fastparquet')
                print(f"✅ 使用fastparquet保存成功")
            except Exception as e2:
                print(f"❌ 所有parquet引擎都失败，保存为CSV: {e2}")
                csv_file = data_dir / f"qingfeng_conversations{suffix_str}_{timestamp}.csv"
                df.to_csv(csv_file, index=False, encoding='utf-8')
                print(f"✅ CSV文件保存成功: {csv_file}")
                return
        
        # 保存元数据
        metadata = {
            'dataset_info': {
                'name': '晴风日常对话数据集',
                'description': '使用晴风人格的LLaMA模型生成的1000个日常对话场景',
                'total_conversations': len(test_results),
                'successful_conversations': sum(1 for r in test_results if r.get('success', False)),
                'model': 'llama_4_maverick',
                'persona': '晴风 - 温暖陪伴型AI',
                'generation_date': datetime.now().isoformat()
            },
            'columns': {
                'scenario_id': '场景编号',
                'system_prompt': '系统提示词(晴风人格设定)',
                'prompt': '用户输入(日常对话场景)',
                'response': '晴风的回复',
                'success': '对话是否成功',
                'inference_time': '推理耗时(秒)',
                'total_time': '总耗时(秒)',
                'timestamp': '生成时间戳',
                'model': '使用的模型',
                'temperature': '生成温度参数',
                'max_tokens': '最大token数'
            },
            'statistics': {
                'avg_inference_time': sum(r.get('inference_time', 0) for r in test_results if r.get('success', False)) / max(1, sum(1 for r in test_results if r.get('success', False))),
                'success_rate': sum(1 for r in test_results if r.get('success', False)) / len(test_results) if test_results else 0,
                'total_scenarios': len(test_results)
            }
        }
        
        metadata_file = data_dir / f"qingfeng_conversations{suffix_str}_{timestamp}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 测试结果已保存:")
        print(f"   📊 Parquet: {parquet_file}")
        print(f"   📋 元数据: {metadata_file}")
        print(f"   📈 总对话数: {len(test_results)}")
        
    except Exception as e:
        print(f"⚠️  保存测试结果失败: {e}")

def save_test_results(test_results):
    """保存测试结果 - 兼容旧版本"""
    save_test_results_parquet(test_results)

def main():
    parser = argparse.ArgumentParser(description="晴风聊天 - 有温度的AI对话伙伴")
    parser.add_argument("--llama-url", default="http://127.0.0.1:10018/v1/chat/completions", 
                       help="LLaMA模型服务器地址")
    parser.add_argument("--test", action="store_false",
                       help="运行日常生活对话测试")
    parser.add_argument("--chat", action="store_true",
                       help="启动交互式聊天模式")
    parser.add_argument("--workers", type=int, default=50,
                       help="并发进程数 (默认8)")
    parser.add_argument("--start", type=int, default=0,
                       help="起始位置 (默认0)")
    parser.add_argument("--num", type=int, default=None,
                       help="处理数量 (默认全部1000个)")
    
    args = parser.parse_args()
    
    if args.test:
        run_daily_life_tests(workers=args.workers)
    elif args.chat:
        chat = QingfengChat(args.llama_url)
        chat.interactive_chat()
    else:
        # 默认运行测试
        run_daily_life_tests(workers=args.workers)

if __name__ == "__main__":
    main()
