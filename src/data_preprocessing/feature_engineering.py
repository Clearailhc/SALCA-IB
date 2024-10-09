import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import openai
import json
import time
from datetime import datetime, timedelta
from scipy import stats
from time_series_sample import TimeSeriesSample

# 配置 OpenAI API 密钥和基础URL
openai.api_key = os.getenv("OPENAI_API_KEY", "infra-ppt-creativity")
openai.api_base = os.getenv("LLM_API_BASE_URL", "https://api.lingyiwanwu.com/v1")  # 设置基础URL

def load_processed_data(pos_data_dir, neg_data_dir):
    """
    加载处理后的正负样本数据。每个文件被视为一个完整的样本。

    Args:
        pos_data_dir (str): 正样本目录路径。
        neg_data_dir (str): 负样本目录路径。

    Returns:
        list of TimeSeriesSample: 包含所有样本数据的列表。
    """
    def load_samples_from_dir(directory, label):
        samples = []
        for filename in os.listdir(directory):
            if filename.endswith('.csv'):
                file_path = os.path.join(directory, filename)
                df = pd.read_csv(file_path)
                samples.append(TimeSeriesSample.from_dataframe(df, label))
        return samples

    pos_samples = load_samples_from_dir(pos_data_dir, label=1)
    neg_samples = load_samples_from_dir(neg_data_dir, label=0)

    print(f"加载了 {len(pos_samples)} 个正样本")
    print(f"加载了 {len(neg_samples)} 个负样本")
    print(f"总样本数: {len(pos_samples) + len(neg_samples)}")

    all_samples = pos_samples + neg_samples
    
    # 计算正负样本比例
    pos_ratio = len(pos_samples) / len(all_samples)
    neg_ratio = len(neg_samples) / len(all_samples)
    print(f"正样本比例: {pos_ratio:.2%}")
    print(f"负样本比例: {neg_ratio:.2%}")

    return all_samples

def calculate_window_size(hours):
    """
    根据小时数计算时间窗口大小。

    Args:
        hours (float): 所需的小时数。

    Returns:
        int: 对应的时间窗口大小（数据点数量）。
    """
    return int(hours * 6)  # 每小时6个数据点（10分钟间隔）

def preprocess_data(samples, window_size):
    """
    预处理数据，包括处理缺失值、标准化和创建滑动窗口。

    Args:
        samples (list of TimeSeriesSample): 原始样本列表。
        window_size (int): 时间窗口大小（数据点数量）。

    Returns:
        list of TimeSeriesSample: 预处理后的样本列表。
        StandardScaler: 标准化器对象。
        list: 所有特征名称的列表。
    """
    # 获取所有唯一的特征名
    all_feature_names = list(set(feature for sample in samples for feature in sample.feature_names))
    all_feature_names.sort()  # 确保特征顺序一致

    # 找出最大的时间步长
    max_time_steps = max(sample.features.shape[0] for sample in samples)

    # 创建标准化器
    scaler = StandardScaler()

    # 对每个样本进行预处理
    for sample in samples:
        sample.preprocess(all_feature_names, max_time_steps, window_size, scaler)

    return samples, scaler, all_feature_names

def select_features_traditional(X, y, n_features=10):
    """
    使用传统方法进行特征选择。

    Args:
        X (np.ndarray): 特征矩阵，形状为 (n_samples, n_time_steps, n_features)。
        y (np.ndarray): 标签向量。
        n_features (int): 要选择的特征数量。

    Returns:
        np.ndarray, list, SelectKBest: 选择后的特征矩阵、选择的特征索引和选择器对象。
    """
    # 计算每个样本在时间维度上的平均值
    X_mean = X.mean(axis=1)  # 现在 X_mean 的形状是 (n_samples, n_features)

    # 使用互信息回归进行特征选择
    selector = SelectKBest(mutual_info_regression, k=n_features)
    X_new = selector.fit_transform(X_mean, y)
    
    # 获取被选中的特征的索引
    selected_feature_indices = selector.get_support(indices=True)
    
    # 在原始的三维数组上选择特征
    X_selected = X[:, :, selected_feature_indices]

    return X_selected, selected_feature_indices, selector

def select_features_with_llm(X, y, feature_names, memory, window_size):
    """
    使用LLM进行特征选择。

    Args:
        X (np.ndarray): 特征矩阵，形状为 (n_samples, n_time_steps, n_features)。
        y (np.ndarray): 标签向量。
        feature_names (list): 特征名称列表。
        memory (dict): 记忆系统。
        window_size (int): 时间窗口大小。

    Returns:
        np.ndarray, list: 选择后的特征矩阵和选择的特征索引。
    """
    # print(f"在select_features_with_llm中，X shape: {X.shape}")

    # 将所有样本和时间步展平为一个2D数组
    X_flat = X.reshape(-1, X.shape[2])

    # 计算每个特征的方差
    X_var = X_flat.var(axis=0)

    # 计算每个特征的平均值
    X_mean = X_flat.mean(axis=0)

    # 计算每个特征的最大值和最小值
    X_max = X_flat.max(axis=0)
    X_min = X_flat.min(axis=0)

    # 准备特征统计信息
    feature_stats = [
        {
            "name": name,
            "mean": mean,
            "variance": var,
            "max": max_val,
            "min": min_val
        }
        for name, mean, var, max_val, min_val in zip(feature_names, X_mean, X_var, X_max, X_min)
    ]

    # 构建提示信息
    prompt = f"""
    我有以下特征及其统计信息：

    {json.dumps(feature_stats, ensure_ascii=False, indent=2)}

    时间窗口大小: {window_size}

    历史模式：
    {json.dumps(memory.get('long_term_memory', {}).get('historical_patterns', []), ensure_ascii=False)}

    最近的反馈：
    {json.dumps(memory.get('short_term_memory', {}).get('recent_feedback', []), ensure_ascii=False)}

    请基于这些信息，选择对预测目标变量最有用的特征。特征数量不固定，由你根据数据特征和历史反馈自主决定。
    请解释你的选择理由，并说明为什么选择这个数量的特征。

    请按以下格式返回你的选择和解释：

    选择的特征:
    - feature1
    - feature2
    - ...

    解释:
    你的解释文本。

    回复格式：
    选择的特征：
    - feature1
    - feature2
    - ...

    解释：你的解释文本。

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
                {"role": "system", "content": "你是一个数据科学家，擅长特征选择和模型优化。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )
    except Exception as e:
        print(f"调用LLM时出错: {e}")
        return None, []

    # 解析 LLM 响应
    llm_output = response.choices[0].message.content
    selected_features, explanation = parse_llm_output(llm_output)

    if not selected_features:
        print("LLM未能返回有效的特征选择结果，使用传统方法进行选择。")
        return None, []

    # 确保所有选择的特征都在原始特征列表中
    selected_features = [feat for feat in selected_features if feat in feature_names]

    # 根据选择的特征索引进行特征选择
    selected_indices = [feature_names.index(feat) for feat in selected_features]
    X_selected = X[:, :, selected_indices]

    # 更新短期记忆
    memory['short_term_memory']['recent_features'] = selected_features
    memory['short_term_memory']['recent_feedback'].append(explanation)

    print(f"LLM 选择了 {len(selected_features)} 个特征。")
    print(f"选择的特征: {', '.join(selected_features)}")
    print(f"解释：{explanation}")

    return X_selected, selected_indices

def parse_llm_output(llm_output):
    """
    解析LLM的输出，提取选择的特征和解释。

    Args:
        llm_output (str): LLM的输出文本。

    Returns:
        list, str: 择的特征名称列表和选择理由。
    """
    selected_features = []
    explanation = ""
    
    lines = llm_output.split('\n')
    feature_section = False
    explanation_section = False
    
    for line in lines:
        if "选择的特征" in line.strip():
            feature_section = True
            explanation_section = False
        elif "解释" in line.strip():
            feature_section = False
            explanation_section = True
        elif feature_section and line.strip().startswith('-'):
            selected_features.append(line.strip()[1:].strip())
        elif explanation_section:
            explanation += line.strip() + " "

    return selected_features, explanation.strip()

def select_window_size_with_llm(memory):
    """
    利用LLM根据记忆自动选择时间窗口大小。

    Args:
        memory (dict): 记忆系统。

    Returns:
        float: 选择的时间窗口大小（小时）。
    """
    historical_patterns = memory.get('long_term_memory', {}).get('historical_patterns', [])
    recent_feedback = memory.get('short_term_memory', {}).get('recent_feedback', [])
    
    prompt = f"""
    根据以下信息，选择一个合适的时间窗口大小（以小时为单位）：

    历史模式：
    {json.dumps(historical_patterns, ensure_ascii=False, indent=2)}

    最近的反馈：
    {json.dumps(recent_feedback, ensure_ascii=False, indent=2)}

    请考虑以下因素：
    1. 历史模式中出现的时间相关特征
    2. 最近反馈中提到的时间窗口建议
    3. 故障模式的典型持续时间
    4. 数据的采样频率（每10分钟一个数据点）
    5. 原始数据最多只采集了3小时，因此时间窗口不能超过3小时

    请给出一个小时数作为时间窗口大小，可以是小数（精确到0.5小时），并解释你的选择理由。
    时间窗口必须大于等于0.5小时且小于等于3小时。

    回复格式：
    时间窗口大小：X小时
    解释：你的解释文本。
    """

    try:
        client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "infra-ppt-creativity"),
            base_url=os.getenv("LLM_API_BASE_URL", "https://api.lingyiwanwu.com/v1")
        )
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "你是一个数据科学家，擅长时间序列分析和特征工程。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )
    except Exception as e:
        print(f"调用LLM时出错: {e}")
        return 1.5  # 默认返回1.5小时作为备选

    llm_output = response.choices[0].message.content
    window_size, explanation = parse_window_size_output(llm_output)

    print(f"LLM建议的时间窗口大小：{window_size}小时")
    print(f"解释：{explanation}")

    return window_size

def parse_window_size_output(llm_output):
    """
    解析LLM的出，提取时间窗口大小和解释。

    Args:
        llm_output (str): LLM的输出文本。

    Returns:
        float, str: 时间窗口大小（小时）和解释。
    """
    lines = llm_output.split('\n')
    window_size = 1.5  # 默认值
    explanation = ""
    
    for line in lines:
        if line.startswith("时间窗口大小："):
            try:
                window_size = float(line.split("：")[1].split("小时")[0])
            except ValueError:
                print("无法解析时间窗口大小，使用默认值1.5小时")
        elif line.startswith("解释："):
            explanation = line.split("：", 1)[1].strip()

    return window_size, explanation

def select_and_save_features(memory):
    """
    执行特征选择和保存流程。

    Args:
        memory (dict): 记忆系统。
    """
    processed_data_dir = "./data/raw/processed/"
    pos_data_dir = os.path.join(processed_data_dir, "pos")
    neg_data_dir = os.path.join(processed_data_dir, "neg")
    output_selected_data = os.path.join(processed_data_dir, "selected_features.jsonl")

    print("加载数据...")
    samples = load_processed_data(pos_data_dir, neg_data_dir)

    print("使用LLM选择时间窗口大小...")
    window_hours = select_window_size_with_llm(memory)
    window_size = calculate_window_size(window_hours)

    print(f"使用 {window_hours:.1f} 小时的时间窗口（{window_size} 个数据点）")

    print("预处理数据...")
    preprocessed_samples, scaler, all_feature_names = preprocess_data(samples, window_size)

    # 准备特征矩阵和标签向量
    X = np.array([sample.preprocessed_features for sample in preprocessed_samples])
    y = np.array([sample.label for sample in preprocessed_samples])

    print("使用LLM进行特征选择...")
    X_selected, selected_feature_indices = select_features_with_llm(X, y, all_feature_names, memory, window_size)

    if X_selected is None:
        print("LLM特征选择失败，使用传统方法...")
        X_selected, selected_feature_indices, selector = select_features_traditional(X, y)
    else:
        selector = None

    # 获取选中特征的名称
    selected_feature_names = [all_feature_names[i] for i in selected_feature_indices]

    print("保存选择后的特征和相关对象...")
    save_selected_features(preprocessed_samples, selected_feature_indices, selected_feature_names, 
                           scaler, selector, output_selected_data, window_size)

    print("特征工程完成。选择的特征已保存至", output_selected_data)

def save_selected_features(samples, selected_feature_indices, selected_feature_names, scaler, selector, output_path, window_size):
    """
    保存选择后的特征和相关对象。

    Args:
        samples (list of TimeSeriesSample): 预处理后的样本列表。
        selected_feature_indices (list): 选择的特征索引。
        selected_feature_names (list): 选择的特征名称。
        scaler (StandardScaler): 标准化器对象。
        selector (SelectKBest or None): 特征选择器对象。
        output_path (str): 输出路径。
        window_size (int): 时间窗口大小（数据点数量）。
    """
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 保存样本数据为 JSONL 文件
    jsonl_path = os.path.splitext(output_path)[0] + '.jsonl'
    with open(jsonl_path, 'w') as f:
        for sample in samples:
            # 只保存选中的特征
            sample.preprocessed_features = sample.preprocessed_features[:, :, selected_feature_indices]
            sample.feature_names = selected_feature_names
            f.write(sample.to_json() + '\n')
    print(f"选择的特征数据已保存至 {jsonl_path}")

    # 保存 metadata
    metadata = {
        'window_size': window_size,
        'selected_features': selected_feature_names
    }
    metadata_path = os.path.join(os.path.dirname(output_path), 'metadata.pkl')
    joblib.dump(metadata, metadata_path)
    print(f"Metadata 已保存至 {metadata_path}")

    # 保存 scaler
    if scaler is not None:
        scaler_path = os.path.join(os.path.dirname(output_path), 'scaler.pkl')
        joblib.dump(scaler, scaler_path)
        print(f"Scaler 已保存至 {scaler_path}")
    else:
        print("没有提供 Scaler 对象，跳过保存")

    # 如果有 selector，也保存它
    if selector is not None:
        selector_path = os.path.join(os.path.dirname(output_path), 'selector.pkl')
        joblib.dump(selector, selector_path)
        print(f"Selector 已保存至 {selector_path}")
    else:
        print("没有提供 Selector 对象，跳过保存")

    print(f"时间窗口大小: {window_size} 数据点（{window_size/6:.1f} 小时）")
    print(f"选择的特征: {', '.join(selected_feature_names)}")

def load_selected_features(input_path):
    """
    加载保存的特征数据并重构为3D格式。

    Args:
        input_path (str): 输入CSV文件路径。

    Returns:
        list of dict: 重构后的样本列表。
        dict: 元数据，包含窗口大小和选择的特征。
    """
    # 加载CSV数据
    df = pd.read_csv(input_path)
    
    # 加载元数据
    metadata = joblib.load(os.path.join(os.path.dirname(input_path), 'metadata.pkl'))
    window_size = metadata['window_size']
    selected_features = metadata['selected_features']

    # 重构样本
    samples = []
    for sample_id in df['sample_id'].unique():
        sample_df = df[df['sample_id'] == sample_id]
        features = sample_df[selected_features].values
        label = sample_df['label'].iloc[0]
        timestamp = sample_df['timestamp'].values
        
        samples.append({
            'features': features,
            'label': label,
            'timestamp': timestamp
        })

    return samples, metadata

if __name__ == "__main__":
    # 定义记忆系统
    memory_system = {
        'long_term_memory': {
            'historical_patterns': [
                {
                    'pattern_id': 1,
                    'description': '高偏置电流导致的设备过热故障模式',
                    'features': [
                        'ib_device_stat_bias_current_c_0',
                        'ib_device_stat_bias_current_c_1',
                        'ib_device_stat_bias_current_c_2',
                        'ib_device_stat_bias_current_c_3',
                        'ib_device_stat_temperature'
                    ],
                    'typical_duration': '1-3小时',
                    'occurrence_rate': 0.04,
                    'recommended_window': '2-3小时'
                },
                {
                    'pattern_id': 2,
                    'description': '接收功率异常波动引发的号质量下降',
                    'features': [
                        'ib_device_stat_rx_power_current_c_0',
                        'ib_device_stat_rx_power_current_c_1',
                        'ib_device_stat_rx_power_current_c_2',
                        'ib_device_stat_rx_power_current_c_3'
                    ],
                    'typical_duration': '1-2小时',
                    'occurrence_rate': 0.03,
                    'recommended_window': '1.5-2小时'
                },
                {
                    'pattern_id': 3,
                    'description': '发送功率过高导致的信号干扰问题',
                    'features': [
                        'ib_device_stat_tx_power_current_c_0',
                        'ib_device_stat_tx_power_current_c_1',
                        'ib_device_stat_tx_power_current_c_2',
                        'ib_device_stat_tx_power_current_c_3'
                    ],
                    'typical_duration': '30分钟-2小时',
                    'occurrence_rate': 0.02,
                    'recommended_window': '1-1.5小时'
                }
            ],
            'model_performance': {
                'xgboost': {'accuracy': 0.82, 'f1_score': 0.78, 'optimal_window': '2.5小时'},
                'lstm': {'accuracy': 0.85, 'f1_score': 0.80, 'optimal_window': '2.5小时'},
                'gru': {'accuracy': 0.84, 'f1_score': 0.79, 'optimal_window': '2.5小时'},
                'cnn': {'accuracy': 0.80, 'f1_score': 0.75, 'optimal_window': '2小时'},
                'transformer_encoder': {'accuracy': 0.86, 'f1_score': 0.81, 'optimal_window': '2.5小时'}
            }
        },
        'short_term_memory': {
            'recent_features': [
                'ib_device_stat_bias_current_c_0',
                'ib_device_stat_bias_current_c_1',
                'ib_device_stat_bias_current_c_2',
                'ib_device_stat_bias_current_c_3',
                'ib_device_stat_rx_power_current_c_0',
                'ib_device_stat_tx_power_current_c_0',
                'ib_device_stat_temperature'
            ],
            'recent_feedback': [
                '最近的实验表明，2.5小时的时间窗口在捕捉大多数故障模式方面表现良好。',
                '对于某些快速变化的特征，如接收功率波动，1.5-2小时的窗口可能更合适。',
                '考虑到数据采集限制，3小时的窗口能够捕捉到最长的可能趋势。',
                '1小时的窗口在捕捉短期波动方面表现出色，特别是对于发送功率相关的问题。',
                '综合考虑各种故障模式，2-2.5小时的窗口似乎能够在捕捉短期波动和长期趋势之间取得良好的平衡。'
            ],
            'window_size_history': [
                {'timestamp': '2023-06-01', 'window_size': 1.5, 'performance': 'Good'},
                {'timestamp': '2023-06-15', 'window_size': 2.0, 'performance': 'Better'},
                {'timestamp': '2023-07-01', 'window_size': 2.5, 'performance': 'Best so far'},
                {'timestamp': '2023-07-15', 'window_size': 3.0, 'performance': 'Good, but at the limit'}
            ]
        }
    }

    # 执行特征选择和保存
    select_and_save_features(memory_system)