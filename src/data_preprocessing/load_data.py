import os
import pandas as pd
import joblib
import json
import openai
import numpy as np
from sklearn.model_selection import train_test_split
from openai import OpenAI
from datetime import datetime
from time_series_sample import TimeSeriesSample

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
        train_ratio (float): 训练集占选定数据的比例，默认为0.8。
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

def select_training_data_with_llm_timestamp(samples, memory):
    """
    利用LLM进行自动训练数据选择基于时间戳，并参考记忆系统。

    Args:
        samples (list of TimeSeriesSample): 原始数据样本列表。
        memory (dict): 记忆系统，包括 'long_term_memory' 和 'short_term_memory'。

    Returns:
        list, list: 训练集样本列表和测试集样本列表。
    """
    # 构建提示信息，包含长期记忆中的模式和短期记忆中的反馈
    historical_patterns = memory.get('long_term_memory', {}).get('historical_patterns', [])
    recent_feedback = memory.get('short_term_memory', {}).get('recent_feedback', [])

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

    历史模式：
    {json.dumps(historical_patterns, ensure_ascii=False)}
    
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
    memory['short_term_memory']['recent_training_selection'] = {
        'time_period': {
            '开始时间': datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'),
            '结束时间': datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
        },
        'train_ratio': train_ratio
    }
    memory['short_term_memory']['recent_feedback'].append(explanation)

    print(f"选择的时间段: {datetime.fromtimestamp(start_time)} 到 {datetime.fromtimestamp(end_time)}")
    print(f"训练集比例: {train_ratio}")
    print(f"解释：{explanation}")

    # 打印总样本数、训练集样本数和测试集样本数
    print(f"总样本数: {len(selected_samples)}")
    print(f"训练集样本数: {len(train_samples)}")
    print(f"测试集样本数: {len(test_samples)}")

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
        np.ndarray, list, object: 选择后的特征矩阵、选择的特征索引和选择器对象。
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

def select_and_save_training_data(memory=None):
    """
    执行训练数据选择和保存流程。

    Args:
        memory (dict, optional): 记忆系统，包括 'long_term_memory' 和 'short_term_memory'。默认为 None。

    Raises:
        ValueError: 当使用 LLM 选择训练时间但 memory 为空时抛出。
    """
    data_path = "./data/raw/processed/selected_features.jsonl"  # 数据文件路径
    output_dir = "./data/raw/processed/training_data/"

    print("加载数据...")
    samples = load_data(data_path)

    # 选择训练数据的方法：'llm_time'
    selection_method = 'llm_time'

    if selection_method == 'llm_time':
        print("使用LLM自动选择训练和测试数据的时间段和比例...")
        if memory is None:
            raise ValueError("使用 LLM 选择训练时间时，memory 参数不能为空。")
        train_samples, test_samples = select_training_data_with_llm_timestamp(samples, memory)
    else:
        raise ValueError("不支持的数据选择方法。")

    print("保存选择后的训练和测试数据...")
    save_selected_training_data(train_samples, test_samples, output_dir)

    print("训练数据选择完成。")

if __name__ == "__main__":
    # 示例记忆系统结构（参考真实样本数据）
    memory_system = {
        'long_term_memory': {
            'historical_patterns': [
                {
                    'pattern_id': 1,
                    'description': '夏季高温导致的设备过热故障模式',
                    'features': [
                        'ib_device_stat_bias_current_c_0',
                        'ib_device_stat_bias_current_c_1',
                        'ib_device_stat_bias_current_c_2',
                        'ib_device_stat_bias_current_c_3',
                        'ib_device_stat_temperature'
                    ],
                    'occurrence_rate': 0.05,
                    'typical_period': '7月中旬至8月中旬'
                },
                {
                    'pattern_id': 2,
                    'description': '雷雨季节信号质量波动',
                    'features': [
                        'ib_device_stat_rx_power_current_c_0',
                        'ib_device_stat_rx_power_current_c_1',
                        'ib_device_stat_rx_power_current_c_2',
                        'ib_device_stat_rx_power_current_c_3'
                    ],
                    'occurrence_rate': 0.03,
                    'typical_period': '6月下旬至8月上旬'
                },
                {
                    'pattern_id': 3,
                    'description': '设备维护后的性能波动',
                    'features': [
                        'ib_device_stat_tx_power_current_c_0',
                        'ib_device_stat_tx_power_current_c_1',
                        'ib_device_stat_tx_power_current_c_2',
                        'ib_device_stat_tx_power_current_c_3'
                    ],
                    'occurrence_rate': 0.02,
                    'typical_period': '每月初'
                }
            ],
            'model_performance': {
                'xgboost': {'accuracy': 0.83, 'f1_score': 0.79, 'last_updated': '2024-08-15'},
                'lstm': {'accuracy': 0.86, 'f1_score': 0.82, 'last_updated': '2024-08-15'},
                'gru': {'accuracy': 0.85, 'f1_score': 0.81, 'last_updated': '2024-08-15'},
                'transformer_encoder': {'accuracy': 0.87, 'f1_score': 0.83, 'last_updated': '2024-08-15'}
            },
            'data_characteristics': {
                'total_samples': 65000,  # 假设的样本数量
                'positive_ratio': 0.15,  # 假设的正样本比例
                'negative_ratio': 0.85,  # 假设的负样本比例
                'typical_daily_samples': 1000  # 假设的每日平均样本数
            }
        },
        'short_term_memory': {
            'recent_training_selection': {
                'time_period': {
                    '开始时间': '2024-07-01',
                    '结束时间': '2024-08-15'
                },
                'train_ratio': 0.8
            },
            'recent_feedback': [
                '选择7月至8月中旬的数据作为训练集，以捕捉夏季高温和雷雨季节的特��。',
                '保留0.2的数据作为测试集，用于评估模型在最近一周数据上的表现。',
                '考虑增加对设备维护后性能波动的关注，可能需要更细粒度的时间划分。'
            ]
        }
    }

    # 调用函数并传入示例记忆系统
    select_and_save_training_data(memory_system)