import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from memory.memory import MemorySystem

# 配置 OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY", "infra-ppt-creativity")
openai.api_base = os.getenv("LLM_API_BASE_URL", "https://api.lingyiwanwu.com/v1")

def load_processed_data(pos_data_dir, neg_data_dir):
    """
    加载处理后的正负样本数据。

    Args:
        pos_data_dir (str): 正样本目录路径。
        neg_data_dir (str): 负样本目录路径。

    Returns:
        list of TimeSeriesSample: 包含所有样本数据的列表。

    Note:
        每个CSV文件被视为一个完整的样本。
        函数会打印加载的样本数量和正负样本比例。
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

    Note:
        假设每小时有6个数据点（10分钟间隔）。
    """
    return int(hours * 6)

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
    使用传统方法（互信息回归）进行特征选择。

    Args:
        X (np.ndarray): 特征矩阵，形状为 (n_samples, n_time_steps, n_features)。
        y (np.ndarray): 标签向量。
        n_features (int): 要选择的特征数量。

    Returns:
        np.ndarray: 选择后的特征矩阵。
        list: 选择的特征引。
        SelectKBest: 选择器对象。
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

def select_features_with_llm(X, y, feature_names, memory_system, window_size):
    """
    使用LLM进行特征选择。

    Args:
        X (np.ndarray): 特征矩阵，形状为 (n_samples, n_time_steps, n_features)。
        y (np.ndarray): 标签向量。
        feature_names (list): 特征名称列表。
        memory_system (MemorySystem): 记忆系统。
        window_size (int): 时间窗口大小。

    Returns:
        np.ndarray: 选择后的特征矩阵。
        list: 选择的特征索引。
    """
    # 将所有样本和时间步展平为一个2D数组
    X_flat = X.reshape(-1, X.shape[2])

    # 计算每个特征的统计信息
    X_var = X_flat.var(axis=0)
    X_mean = X_flat.mean(axis=0)
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

    # 获取历史故障模式
    historical_patterns = memory_system.get_memory('long_term', 'failure_patterns', 'feature_correlations')
    
    # 获取最近的反馈
    recent_feedback = memory_system.get_memory('short_term', 'feedback', 'recent')

    # 构建提示信息
    prompt = f"""
    我有以下特征及其统计信息：

    {json.dumps(feature_stats, ensure_ascii=False, indent=2)}

    时间窗口大小: {window_size}

    历史故障模式：
    {json.dumps(historical_patterns, ensure_ascii=False, indent=2)}

    最近的反馈：
    {json.dumps(recent_feedback, ensure_ascii=False, indent=2)}

    请基于这些信息，选择对预测目标变量最有用的特征。特征数量不固定，由你根据数据特征、历史故障模式和最近反馈自主决定。
    请解释你的选择理由，并说明为什么选择这个数量的特征。

    请按以下格式返回你的选择和解释：

    选择的特征:
    - feature1
    - feature2
    - ...

    解释:
    你的解释文本。
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
    selected_features, explanation = parse_llm_output(llm_output, feature_names)

    if not selected_features:
        print("LLM未能返回有效的特征选择结果，使用传统方法进行选择。")
        return None, []

    # 确保所有选择的特征都在原始特征列表中
    selected_features = [feat for feat in selected_features if feat in feature_names]

    # 根据选择的特征索引进行特征选择
    selected_indices = [feature_names.index(feat) for feat in selected_features]
    X_selected = X[:, :, selected_indices]

    # 更新短期记忆
    memory_system.update_memory('short_term', 'recent_features', 'data', selected_features)
    memory_system.update_memory('short_term', 'feature_selection_explanation', 'data', explanation)

    print(f"LLM 选择了 {len(selected_features)} 个特征。")
    print(f"选择的特征: {', '.join(selected_features)}")
    print(f"解释：{explanation}")

    return X_selected, selected_indices

def parse_llm_output(llm_output, feature_names):
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
            feature = line.strip()[1:].strip()
            if feature in feature_names:
                selected_features.append(feature)
        elif explanation_section:
            explanation += line.strip() + " "

    return selected_features, explanation.strip()

def select_window_size_with_llm(memory_system):
    """
    利用LLM根据记忆自动选择时间窗口大小。

    Args:
        memory_system (MemorySystem): 记忆系统。

    Returns:
        float: 选择的时间窗口大小（小时）。
    """
    historical_patterns = memory_system.get_memory('long_term', 'failure_patterns', 'feature_correlations')
    recent_feedback = memory_system.get_memory('short_term', 'feedback', 'recent')
    model_performance = memory_system.get_memory('long_term', 'model_performance', 'trends')
    
    prompt = f"""
    根据以下信息，选择一个合适的时间窗口大小（以小时为单位）：

    历史故障模式：
    {json.dumps(historical_patterns, ensure_ascii=False, indent=2)}

    最近的反馈：
    {json.dumps(recent_feedback, ensure_ascii=False, indent=2)}

    模型性能：
    {json.dumps(model_performance, ensure_ascii=False, indent=2)}

    请考虑以下因素：
    1. 历史故障模式中提到的典型持续时间和推荐窗口
    2. 最近反馈中提到的时间窗口建议
    3. 不同模型在不同窗口大小下的性能表现
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

    # 更新短期记忆
    memory_system.update_memory('short_term', 'window_size', 'current', window_size)
    memory_system.update_memory('short_term', 'window_size', 'explanation', explanation)

    print(f"LLM建议的时间窗口大小：{window_size}小时")
    print(f"解释：{explanation}")

    return window_size

def parse_window_size_output(llm_output):
    """
    解析LLM的输出，提取时间窗口大小和解释。

    Args:
        llm_output (str): LLM的输出文本。

    Returns:
        float: 时间窗口大小（小时）。
        str: 解释。
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

def select_and_save_features(memory_path=None):
    """
    执行特征选择和保存流程。

    Args:
        memory_path (str, optional): 记忆系统的JSON文件路径。
                                     如果未提供，将使用默认路径。

    Note:
        这个函数整合了整个特征工程流程，包括数据加载、预处理、特征选择和保存。
    """
    processed_data_dir = "./data/raw/processed/"
    pos_data_dir = os.path.join(processed_data_dir, "pos")
    neg_data_dir = os.path.join(processed_data_dir, "neg")
    output_selected_data = os.path.join(processed_data_dir, "selected_features.jsonl")

    # 使用提供的路径或默认路径载记忆系统
    default_memory_path = './src/memory/memory_test.json'
    memory_path = memory_path or default_memory_path
    
    try:
        memory_system = MemorySystem.load_from_json(memory_path)
        print(f"成功从 {memory_path} 加载记忆系统")
    except FileNotFoundError:
        print(f"警告：未找到记忆文件 {memory_path}，将创建新的记忆系统")
        memory_system = MemorySystem()

    print("加载数据...")
    samples = load_processed_data(pos_data_dir, neg_data_dir)

    print("使用LLM选择时间窗口大小...")
    window_hours = select_window_size_with_llm(memory_system)
    window_size = calculate_window_size(window_hours)

    print(f"使用 {window_hours:.1f} 小时的时间窗口（{window_size} 个数据点）")

    print("预处理数据...")
    preprocessed_samples, scaler, all_feature_names = preprocess_data(samples, window_size)

    # 准备特征矩阵和标签向量
    X = np.array([sample.preprocessed_features for sample in preprocessed_samples])
    y = np.array([sample.label for sample in preprocessed_samples])

    print("使用LLM进行特征选择...")
    X_selected, selected_feature_indices = select_features_with_llm(X, y, all_feature_names, memory_system, window_size)

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

    print("特征选择完成。选择的特征已保存至", output_selected_data)

    # 更新记忆系统
    memory_system.update_memory('short_term', 'data_characteristics', 'current', {
        'window_size': window_size,
        'selected_features': selected_feature_names
    })
    memory_system.save_to_json(memory_path)
    print(f"更新后的记忆系统已保存至 {memory_path}")

    # 执行记忆整合
    memory_system.consolidate_memory()
    print("记忆整合完成")

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

    Note:
        保存的内容包括样本数据（JSONL格式）、metadata、scaler和selector（如果有）。
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

    Note:
        加载的数据包括样本特征、标签和时间戳。
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
    # 执行特征选择和保存
    # 可以选择性地传入记忆文件路径
    # select_and_save_features('./path/to/custom/memory.json')
    select_and_save_features()
