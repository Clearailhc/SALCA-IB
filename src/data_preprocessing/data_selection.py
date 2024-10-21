import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
from data_preprocessing.time_series_sample import TimeSeriesSample
from memory.memory import MemorySystem
import openai

# 配置 OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY", "infra-ppt-creativity")
openai.api_base = os.getenv("LLM_API_BASE_URL", "https://api.lingyiwanwu.com/v1")

def load_data(data_path):
    """加载数据集。"""
    samples = []
    with open(data_path, 'r') as f:
        for line in f:
            sample = TimeSeriesSample.from_json(line.strip())
            samples.append(sample)
    return samples

def preprocess_data(data):
    """预处理数据，包括处理缺失值和标准化。"""
    # 处理缺失值
    data = data.dropna()

    # 分离特征和标签
    X = data.drop(['timestamp', 'label'], axis=1)
    y = data['label']

    # 标准化特征
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

def select_training_data_with_llm(samples, memory_system):
    """利用LLM进行自动训练数据选择，并参考记忆系统。"""
    # 获取相关记忆
    context = ['failure_patterns', 'model_performance', 'feedback']
    relevant_memory = memory_system.get_relevant_memory(context)

    # 构建提示信息
    historical_patterns = relevant_memory.get('long_term', {}).get('failure_patterns', [])
    model_performance = relevant_memory.get('long_term', {}).get('model_performance', [])
    recent_feedback = relevant_memory.get('short_term', {}).get('feedback', [])

    # 获取数据的时间范围
    all_timestamps = [sample.timestamp[0] for sample in samples] + [sample.timestamp[-1] for sample in samples]
    min_time = min(all_timestamps)
    max_time = max(all_timestamps)

    # 添加数据分布信息
    label_distribution = {0: 0, 1: 0}
    for sample in samples:
        label_distribution[sample.label] += 1
    total_samples = len(samples)
    label_distribution = {k: v / total_samples for k, v in label_distribution.items()}
    data_distribution_info = f"标签分布: {json.dumps(label_distribution, ensure_ascii=False)}"

    prompt = f"""
    我有一个数据集用于网络故障预测，包含以下特征：
    {', '.join(samples[0].feature_names)}
    
    数据的时间范围是从 {datetime.fromtimestamp(min_time)} 到 {datetime.fromtimestamp(max_time)}。

    历史故障模式：
    {json.dumps(historical_patterns, ensure_ascii=False)}
    
    模型性能：
    {json.dumps(model_performance, ensure_ascii=False)}
    
    最近的反馈：
    {json.dumps(recent_feedback, ensure_ascii=False)}
    
    数据分布信息：
    {data_distribution_info}
    
    请基于这些信息，包括数据分布，选择最有助于训练模型的时间段和训练集比例。选择的标准可以包括数据的多样性、代表性和与历史模式的关联性。
    请解释你的选择理由，并说明选择的时间段和训练集比例。

    请严格按以下格式返回你的选择和解释：

    选择的时间段:
    - 开始时间: YYYY-MM-DD HH:MM:SS
    - 结束时间: YYYY-MM-DD HH:MM:SS

    训练集比例: 0.X

    解释:
    你的详细解释文本。请提供充分的理由，解释为什么选择这个时间段和训练集比例。
    """

    # 调用 LLM 获取建议
    try:
        client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "infra-ppt-creativity"),
            base_url=os.getenv("LLM_API_BASE_URL", "https://api.lingyiwanwu.com/v1")
        )
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "你是一个数据科学家，擅长数据选择和模型优化。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )
    except Exception as e:
        print(f"调用LLM时出错: {e}")
        return None, None

    # 解析 LLM 响应
    llm_output = response.choices[0].message.content
    time_period, train_ratio, explanation = parse_llm_output(llm_output)

    if time_period is None or train_ratio is None:
        print("LLM未能返回有效的数据选择结果，使用默认方法进行选择。")
        start_time = min(sample.timestamp[0] for sample in samples)
        end_time = max(sample.timestamp[-1] for sample in samples)
        train_ratio = 0.8
    else:
        start_time = int(pd.to_datetime(time_period['开始时间']).timestamp())
        end_time = int(pd.to_datetime(time_period['结束时间']).timestamp())

    # 根据选择的时间段筛选样本
    selected_samples = [
        sample for sample in samples
        if start_time <= sample.timestamp[0] <= end_time
    ]

    # 随机划分训练集和测试集
    from sklearn.model_selection import train_test_split
    train_samples, test_samples = train_test_split(selected_samples, train_size=train_ratio, random_state=42)

    # 更新短期记忆
    selection_info = {
        'time_period': {
            '开始时间': datetime.fromtimestamp(start_time).isoformat(),
            '结束时间': datetime.fromtimestamp(end_time).isoformat()
        },
        'train_ratio': train_ratio,
        'sample_counts': {
            '总样本数': len(samples),
            '选择的样本数': len(selected_samples),
            '训练集样本数': len(train_samples),
            '测试集样本数': len(test_samples)
        },
        'explanation': explanation
    }
    memory_system.update_memory('short_term', 'data_selection', selection_info)

    return train_samples, test_samples

def parse_llm_output(llm_output):
    """解析LLM输出，提取时间段、训练集比例和解释。"""
    lines = llm_output.split('\n')
    time_period = {}
    train_ratio = None
    explanation = ""
    parsing_explanation = False

    for line in lines:
        line = line.strip()
        if line.startswith("- 开始时间:"):
            time_period['开始时间'] = line.split(":")[1].strip()
        elif line.startswith("- 结束时间:"):
            time_period['结束时间'] = line.split(":")[1].strip()
        elif line.startswith("训练集比例:"):
            train_ratio = float(line.split(":")[1].strip())
        elif line.startswith("解释:"):
            parsing_explanation = True
        elif parsing_explanation:
            explanation += line + " "

    return time_period, train_ratio, explanation.strip()

def save_selected_data(train_samples, test_samples, output_path):
    """保存选择的训练和测试数据。"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    train_count = 0
    test_count = 0
    
    with open(output_path, 'w') as f:
        for sample in train_samples + test_samples:
            sample_dict = {
                'timestamp': sample.timestamp.tolist(),
                'features': sample.features.tolist(),
                'label': int(sample.label),
                'split': 'train' if sample in train_samples else 'test'
            }
            json.dump(sample_dict, f, ensure_ascii=False)
            f.write('\n')
            
            if sample in train_samples:
                train_count += 1
            else:
                test_count += 1
    
    print(f"选择的数据已保存至 {output_path}")
    print(f"训练集样本数：{train_count}")
    print(f"测试集样本数：{test_count}")
    print(f"总样本数：{train_count + test_count}")

def select_and_save_data(memory_path=None):
    """主函数：选择并保存数据。"""
    data_path = "data/raw/processed/selected_features.jsonl"
    output_selected_data = "./data/raw/processed/selected_samples.jsonl"

    default_memory_path = './src/memory/memory_test.json'
    memory_path = memory_path or default_memory_path
    
    try:
        memory_system = MemorySystem.load_from_json(memory_path)
        print(f"成功从 {memory_path} 加载记忆系统")
    except FileNotFoundError:
        print(f"警告：未找到记忆文件 {memory_path}，将创建新的记忆系统")
        memory_system = MemorySystem()
    except json.JSONDecodeError:
        print(f"警告：记忆文件 {memory_path} 为空或包含无效的JSON数据，将创建新的记忆系统")
        memory_system = MemorySystem()

    print("加载数据...")
    samples = load_data(data_path)

    print("使用LLM选择训练数据...")
    train_samples, test_samples = select_training_data_with_llm(samples, memory_system)

    if train_samples is None or test_samples is None:
        print("LLM数据选择失败，使用默认方法...")
        # 实现默认的数据选择方法
        pass
    else:
        print("保存选择的数据...")
        save_selected_data(train_samples, test_samples, output_selected_data)

    memory_system.save_to_json(memory_path)
    print(f"更新后的记忆系统已保存至 {memory_path}")

    memory_system.consolidate_memory()
    print("记忆整合完成")

if __name__ == "__main__":
    select_and_save_data()
