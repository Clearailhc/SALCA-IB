import os
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler
import joblib
import openai
import json
import time
import numpy as np

# 配置 OpenAI API 密钥和基础URL
openai.api_key = os.getenv("OPENAI_API_KEY", "infra-ppt-creativity")
openai.api_base = os.getenv("LLM_API_BASE_URL", "https://api.lingyiwanwu.com/v1")  # 设置基础URL

def load_processed_data(pos_data_dir, neg_data_dir):
    """
    加载处理后的正负样本数据。

    Args:
        pos_data_dir (str): 正样本目录路径。
        neg_data_dir (str): 负样本目录路径。

    Returns:
        pd.DataFrame: 合并后的数据集。
    """
    pos_files = [os.path.join(pos_data_dir, f) for f in os.listdir(pos_data_dir) if f.endswith('.csv')]
    neg_files = [os.path.join(neg_data_dir, f) for f in os.listdir(neg_data_dir) if f.endswith('.csv')]

    pos_data = pd.concat([pd.read_csv(f) for f in pos_files], ignore_index=True)
    neg_data = pd.concat([pd.read_csv(f) for f in neg_files], ignore_index=True)

    pos_data['label'] = 1
    neg_data['label'] = 0

    data = pd.concat([pos_data, neg_data], ignore_index=True)
    return data

def preprocess_data(data):
    """
    预处理数据，包括处理缺失值和标准化。

    Args:
        data (pd.DataFrame): 原始数据集。

    Returns:
        np.ndarray, pd.Series, StandardScaler: 特征矩阵、标签向量和标准化器。
    """
    # 处理缺失值
    data = data.dropna()

    # 分离特征和标签
    X = data.drop(['timestamp', 'label'], axis=1)
    y = data['label']

    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

def select_features_traditional(X, y, method='f_regression', k=10):
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

def select_features_with_llm(X, y, feature_names, memory):
    """
    利用LLM进行自动特征选择，并参考记忆系统。LLM将根据数据特征和历史反馈自动决定选择的特征数量。

    Args:
        X (np.ndarray): 特征矩阵。
        y (pd.Series): 标签向量。
        feature_names (list): 原始特征名称列表。
        memory (dict): 记忆系统，包括 'long_term_memory' 和 'short_term_memory'。

    Returns:
        np.ndarray, list: 选择后的特征矩阵和特征索引。
    """
    # 计算基本的统计信息
    df = pd.DataFrame(X, columns=feature_names)
    
    # 移除标准差为零的列
    std_dev = df.std()
    valid_columns = std_dev[std_dev != 0].index
    df = df[valid_columns]
    
    # 重新计算相关性，忽略 NaN 值
    correlation = df.corrwith(pd.Series(y), method='pearson').abs()
    correlation = correlation.sort_values(ascending=False)
    
    # 移除 NaN 值
    correlation = correlation.dropna()

    # 构建提示信息，包含长期记忆中的模式和短期记忆中的反馈
    historical_patterns = memory.get('long_term_memory', {}).get('historical_patterns', [])
    recent_feedback = memory.get('short_term_memory', {}).get('recent_feedback', [])
    
    prompt = f"""
    我有以下特征及其与目标变量的相关性：

    {correlation.to_string()}

    历史模式：
    {json.dumps(historical_patterns, ensure_ascii=False)}

    最近的反馈：
    {json.dumps(recent_feedback, ensure_ascii=False)}

    请基于这些信息，选择对预测目标变量最有用的特征。特征数量不固定，由你根据数据特征和历史反馈自主决定。
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
    selected_features, explanation = parse_llm_output(llm_output)

    if not selected_features:
        print("LLM未能返回有效的特征选择结果，使用传统方法进行选择。")
        return None, []

    # 确保所有选择的特征都在原始特征列表中
    selected_features = [feat for feat in selected_features if feat in feature_names]

    # 根据选择的特征索引进行特征选择
    selected_indices = [feature_names.index(feat) for feat in selected_features]
    X_selected = X[:, selected_indices]

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
        list, str: ��择的特征名称列表和选择理由。
    """
    selected_features = []
    explanation = ""
    
    lines = llm_output.split('\n')
    feature_section = False
    explanation_section = False
    
    for line in lines:
        if line.strip() == "选择的特征:":
            feature_section = True
            explanation_section = False
        elif line.strip() == "解释:":
            feature_section = False
            explanation_section = True
        elif feature_section and line.strip().startswith('-'):
            selected_features.append(line.strip()[1:].strip())
        elif explanation_section:
            explanation += line.strip() + " "

    return selected_features, explanation.strip()

def select_features_with_llm_feedback(X, y, feature_names, memory, max_iterations=3):
    """
    利用LLM进行自动特征选择，并根据反馈迭代更新选择，参考记忆系统。

    Args:
        X (np.ndarray): 特征矩阵。
        y (pd.Series): 标签向量。
        feature_names (list): 原始特征名称列表。
        memory (dict): 记忆系统，包括 'long_term_memory' 和 'short_term_memory'。
        max_iterations (int): 最大迭代次数。

    Returns:
        np.ndarray, list: 选择后的特征矩阵和特征索引。
    """
    for iteration in range(max_iterations):
        print(f"LLM 特征选择迭代 {iteration + 1}/{max_iterations}...")
        X_selected, selected_indices = select_features_with_llm(X, y, feature_names, memory)
        
        if X_selected is None:
            print("LLM选择失败，退出迭代。")
            break
        
        selected_features = [feature_names[idx] for idx in selected_indices]
        
        # 更新记忆系统的短期记忆
        if 'short_term_memory' not in memory:
            memory['short_term_memory'] = {'recent_features': [], 'recent_feedback': []}
        memory['short_term_memory']['recent_features'] = selected_features
        
        # 检查是否有显著变化
        if iteration > 0 and set(selected_features) == set(previous_features):
            print("特征选择没有显著变化，结束迭代。")
            break
        
        previous_features = selected_features

    if not selected_features:
        print("未能通过LLM选择到特征，使用传统方法进行选择。")
        return select_features_traditional(X, y, method='f_regression', k=10)[:2]  # 返回X_new, selected_features

    return X_selected, selected_indices

def save_selected_features(data, selected_feature_indices, scaler, selector, output_path):
    """
    保存选择后的特征和相关对象。

    Args:
        data (pd.DataFrame): 原始数据集。
        selected_feature_indices (list): 选择的特征索引。
        scaler (StandardScaler): 标准化器对象。
        selector (SelectKBest or None): 特征选择器对象。
        output_path (str): 输出路径。
    """
    selected_columns = data.drop(['timestamp', 'label'], axis=1).columns[selected_feature_indices]
    data_selected = data[['label'] + list(selected_columns)]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data_selected.to_csv(output_path, index=False)

    # 保存 scaler 和 selector
    joblib.dump(scaler, os.path.join(os.path.dirname(output_path), 'scaler.pkl'))
    if selector is not None:
        joblib.dump(selector, os.path.join(os.path.dirname(output_path), 'feature_selector.pkl'))

def main(memory):
    """
    主函数，执行特征工程流程。

    Args:
        memory (dict): 记忆系统，包括 'long_term_memory' 和 'short_term_memory'。
    """
    processed_data_dir = "./data/raw/processed/"
    pos_data_dir = os.path.join(processed_data_dir, "pos")
    neg_data_dir = os.path.join(processed_data_dir, "neg")
    output_selected_data = os.path.join(processed_data_dir, "selected_features.csv")

    print("加载数据...")
    data = load_processed_data(pos_data_dir, neg_data_dir)

    print("预处理数据...")
    X, y, scaler = preprocess_data(data)
    feature_names = data.drop(['timestamp', 'label'], axis=1).columns.tolist()

    # 选择特征的方法：'traditional', 'llm', 'llm_feedback'
    selection_method = 'llm_feedback'  # 可以根据需要切换

    if selection_method == 'traditional':
        print("使用传统方法选择特征...")
        X_selected, selected_feature_indices, selector = select_features_traditional(X, y, method='f_regression', k=10)
    elif selection_method == 'llm':
        print("使用LLM自动选择特征...")
        X_selected, selected_feature_indices = select_features_with_llm(X, y, feature_names, memory)
        selector = None  # 如果需要保存LLM选择器，可自定义保存逻辑
    elif selection_method == 'llm_feedback':
        print("使用LLM自动选择特征并根据反馈更新...")
        X_selected, selected_feature_indices = select_features_with_llm_feedback(X, y, feature_names, memory, max_iterations=3)
        selector = None  # LLM选择器不保存
    else:
        raise ValueError("不支持的特征选择方法。")

    print("保存选择后的特征和相关对象...")
    save_selected_features(data, selected_feature_indices, scaler, selector, output_selected_data)

    print("特征工程完成。选择的特征已保存至", output_selected_data)

if __name__ == "__main__":
    # 示例记忆系统结构（参考真实样本数据）
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
                    'occurrence_rate': 0.04
                },
                {
                    'pattern_id': 2,
                    'description': '接收功率异常波动引发的信号质量下降',
                    'features': [
                        'ib_device_stat_rx_power_current_c_0',
                        'ib_device_stat_rx_power_current_c_1',
                        'ib_device_stat_rx_power_current_c_2',
                        'ib_device_stat_rx_power_current_c_3'
                    ],
                    'occurrence_rate': 0.03
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
                    'occurrence_rate': 0.02
                }
                # 可以根据实际情况添加更多历史模式
            ],
            'model_performance': {
                'xgboost': {'accuracy': 0.82, 'f1_score': 0.78},
                'lstm': {'accuracy': 0.85, 'f1_score': 0.80},
                'gru': {'accuracy': 0.84, 'f1_score': 0.79},
                'cnn': {'accuracy': 0.80, 'f1_score': 0.75},
                'transformer_encoder': {'accuracy': 0.86, 'f1_score': 0.81}
                # 可以根据实际情况添加更多模型的性能数据
            }
        },
        'short_term_memory': {
            'recent_features': [
                'ib_device_stat_bias_current_c_0',
                'ib_device_stat_bias_current_c_1',
                'ib_device_stat_bias_current_c_2',
                'ib_device_stat_bias_current_c_3',
                'ib_device_stat_rx_power_current_c_0',
                'ib_device_stat_tx_power_current_c_0'
                # 最近选择的特征
            ],
            'recent_feedback': [
                '添加特征 ib_device_stat_temperature 以监控设备温度变化。',
                '移除特征 ib_device_stat_wavelength 因为其相关性较低且增加了模型复杂度。',
                '考虑增加 ib_device_stat_rx_power_current_c_1 到 c_3，因为它们可能与信号质量下降有关。',
                '建议保留所有偏置电流特征，因为它们在历史模式中频繁出现。'
                # 最近的反馈信息
            ]
        }
    }

    # 调用主函数并传入示例记忆系统
    main(memory_system)