import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import joblib
import json
import openai
import numpy as np
from sklearn.model_selection import train_test_split
from openai import OpenAI
from datetime import datetime
from data_preprocessing.time_series_sample import TimeSeriesSample
from memory.memory import MemorySystem

# 在文件开头添加这行代码
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "infra-ppt-creativity"),
    base_url=os.getenv("LLM_API_BASE_URL", "https://api.lingyiwanwu.com/v1")
)

def load_data(data_path):
    """
    加载数据集。

    Args:
        data_path (str): 数据文件路径。

    Returns:
        list of TimeSeriesSample: 加载的数据集。
    """
    samples = []
    with open(data_path, 'r') as f:
        for line in f:
            sample = TimeSeriesSample.from_json(line.strip())
            samples.append(sample)
    return samples

def preprocess_data(data):
    """
    预处理数据，包括处理缺失值和标准化。

    Args:
        data (pd.DataFrame): 原始数据集。

    Returns:
        np.ndarray, pd.Series, joblib.StandardScaler: 特征矩阵、标签向量和标准化器。
    """
    # 处理缺失值（如果有的话）
    data = data.dropna()

    # 分离特征和标签
    X = data.drop(['timestamp', 'label'], axis=1)
    y = data['label']

    # 标准化特征
    scaler = joblib.load('src/data_preprocessing/scaler.pkl') if os.path.exists('src/data_preprocessing/scaler.pkl') else None
    if scaler:
        X_scaled = scaler.transform(X)
    else:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, 'src/data_preprocessing/scaler.pkl')

    return X_scaled, y, scaler

def select_training_data_traditional_timestamp(data, start_time, end_time, train_ratio=0.8, random_state=42):
    """
    使用传统方法基于时间戳选择训练和测试数据，并按比例划分。

    Args:
        data (pd.DataFrame): 原始数据集，包含 'timestamp' 和 'label' 列。
        start_time (str or pd.Timestamp): 数据选择的开始时间。
        end_time (str or pd.Timestamp): 数据选择的结束时间。
        train_ratio (float): 训练集占选定数据的比例，认为0.8。
        random_state (int): 随机种子，用于可重复的随机划分。

    Returns:
        np.ndarray, np.ndarray, np.ndarray, np.ndarray: 训练集特征、测试集特征、训练集标签、测试集标签。
    """
    # 确保 start_time 和 end_time 是 pd.Timestamp 对象
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)

    # 选择指定时间段内的数据
    mask = (data['timestamp'] >= start_time) & (data['timestamp'] <= end_time)
    selected_data = data.loc[mask]

    if selected_data.empty:
        raise ValueError("选定的时间段内没有数据，请检查时间范围。")

    # 打印选择的样本数
    print(f"选择的样本数: {len(selected_data)}")

    # 分离特征和标签
    X = selected_data.drop(['timestamp', 'label'], axis=1).values
    y = selected_data['label'].values

    # 按比例划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, random_state=random_state)

    # 打印训练集和测试集的样本数
    print(f"训练集样本数: {len(X_train)}")
    print(f"测试集样本数: {len(X_test)}")

    return X_train, X_test, y_train, y_test

def select_training_data_with_llm_timestamp(samples, memory_system):
    """
    利用LLM进行自动训练数据选择基于时间戳，并参考记忆系统。

    Args:
        samples (list of TimeSeriesSample): 原始数据样本列表。
        memory_system (MemorySystem): 记忆系统实例。

    Returns:
        list, list: 训练集样本列表和测试集样本列表。
    """
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
    
    数据的时间范围是从 {min_time} 到 {max_time}。

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
        client = OpenAI(
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
    time_period, train_ratio, explanation = parse_llm_output_training_timestamp(llm_output)

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
    train_samples, test_samples = train_test_split(selected_samples, train_size=train_ratio, random_state=42)

    # 更新短期记忆
    selection_info = {
        'time_period': {
            '开始时间': datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'),
            '结束时间': datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
        },
        'train_ratio': train_ratio,
        'sample_counts': {
            '总样本数': len(samples),
            '训练集样本数': len(train_samples),
            '测试集样本数': len(test_samples)
        },
        'explanation': explanation
    }

    memory_system.update_memory('short_term', 'recent_training_selection', 'data_selection', selection_info)

    print(f"选择的时间段: {selection_info['time_period']['开始时间']} 到 {selection_info['time_period']['结束时间']}")
    print(f"训练集比例: {selection_info['train_ratio']}")
    print(f"解释：{selection_info['explanation']}")
    print(f"总样本数: {selection_info['sample_counts']['总样本数']}")
    print(f"训练集样本数: {selection_info['sample_counts']['训练集样本数']}")
    print(f"测试集样本数: {selection_info['sample_counts']['测试集样本数']}")

    return train_samples, test_samples

def parse_llm_output_training_timestamp(llm_output):
    """
    解析LLM的输出，提取选择的时间段、训练集比例和解释。

    Args:
        llm_output (str): LLM的输出文本。

    Returns:
        dict, float, str: 选择的时间段、训练集比例和解释。
    """
    time_period = {}
    train_ratio = None
    explanation = ""
    
    lines = llm_output.split('\n')
    explanation_started = False
    
    for line in lines:
        line = line.strip()
        if '开始时间:' in line:
            time_period['开始时间'] = line.split('开始时间:')[-1].strip()
        elif '结束时间:' in line:
            time_period['结束时间'] = line.split('结束时间:')[-1].strip()
        elif '训练集比例:' in line:
            try:
                train_ratio = float(line.split(':')[-1].strip())
            except ValueError:
                print("无法解析训练集比例，将使用默认值。")
        elif '解释:' in line or explanation_started:
            explanation_started = True
            if '解释:' in line:
                explanation += line.split('解释:')[-1].strip() + " "
            else:
                explanation += line + " "

    # 验证时间段是否在有效范围内
    min_date = pd.Timestamp('2024-06-20')
    max_date = pd.Timestamp('2024-08-23')
    start_date = pd.to_datetime(time_period['开始时间'])
    end_date = pd.to_datetime(time_period['结束时间'])

    if start_date < min_date or end_date > max_date:
        print("LLM选择的时间段超出有效范围，将使用默认时间范围。")
        time_period = {'开始时间': '2024-06-20', '结束时间': '2024-08-23'}

    return time_period, train_ratio, explanation.strip()

def select_training_data_traditional(X, y, method='f_regression', k=10):
    """
    使用传统的特征选择方法。

    Args:
        X (np.ndarray): 特征矩阵。
        y (pd.Series): 标签向量。
        method (str): 特征选择方法，默认为 'f_regression'。
        k (int): 选择的特征数量。

    Returns:
        np.ndarray, list, object: 选择后特征矩阵、选择的特征索引和选择器对象。
    """
    from sklearn.feature_selection import SelectKBest, f_regression

    selector = SelectKBest(score_func=f_regression, k=k)
    X_new = selector.fit_transform(X, y)
    selected_features = selector.get_support(indices=True)
    return X_new, selected_features, selector

def save_selected_training_data(train_samples, test_samples, output_dir):
    """
    保存选择后的训练和测试数据。

    Args:
        train_samples (list of TimeSeriesSample): 训练集样本。
        test_samples (list of TimeSeriesSample): 测试集样本。
        output_dir (str): 输出目录路径。
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'train_samples.jsonl'), 'w') as f:
        for sample in train_samples:
            f.write(sample.to_json() + '\n')
    
    with open(os.path.join(output_dir, 'test_samples.jsonl'), 'w') as f:
        for sample in test_samples:
            f.write(sample.to_json() + '\n')
    
    print(f"训练测试数据已保存至 {output_dir}")

def select_and_save_training_data(memory_path=None):
    """
    执行训练数据选择和保存流程。

    Args:
        memory_path (str, optional): 记忆系统的JSON文件路径。
                                     如果未提供，将使用默认路径。
    """
    data_path = "./data/raw/processed/selected_features.jsonl"  # 数据文件路径
    output_dir = "./data/raw/processed/training_data/"

    # 使用提供的路径或默认路径加载记忆系统
    default_memory_path = './src/memory/memory_test.json'
    memory_path = memory_path or default_memory_path
    
    try:
        memory_system = MemorySystem.load_from_json(memory_path)
        print(f"成功从 {memory_path} 加载记忆系统")
    except FileNotFoundError:
        print(f"警告：未找到记忆文件 {memory_path}，将创建新的记忆系统")
        memory_system = MemorySystem()

    print("加载数据...")
    samples = load_data(data_path)

    # 选择训练数据的方法：'llm_time'
    selection_method = 'llm_time'

    if selection_method == 'llm_time':
        print("使用LLM自动选择训练和测试数据的时间段和比例...")
        train_samples, test_samples = select_training_data_with_llm_timestamp(samples, memory_system)
    else:
        raise ValueError("不支持的数据选择方法。")

    print("保存选择后的训练和测试数据...")
    save_selected_training_data(train_samples, test_samples, output_dir)

    print("训练数据选择完成。")

    # 整合记忆
    memory_system.consolidate_memory()

    # 保存更新后的记忆
    memory_system.save_to_json(memory_path)
    print(f"更新后的记忆系统已保存至 {memory_path}")

if __name__ == "__main__":
    # 执行训练数据选择和保存
    # 可以选择性地传入记忆文件路径
    select_and_save_training_data()