import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import json
from datetime import timedelta

class TimeSeriesSample:
    """
    TimeSeriesSample 类用于处理和预处理时间序列数据。

    时间序列数据是按时间顺序索引（或列出）的数据点序列。在这个类中：
    - features: 二维数组，形状为 (时间步数, 特征数)，表示随时间变化的多个特征。
    - label: 整数，表示该时间序列样本的类别或标签。
    - timestamp: 一维数组，与特征数组长度相同，表示每个时间步的时间戳。
    - feature_names: 列表，包含所有特征的名称。
    """

    def __init__(self, timestamp, features, label, feature_names, split=None):
        """
        初始化 TimeSeriesSample 对象。

        Args:
            timestamp (np.array): 一维数组，表示时间戳。
            features (np.array): 二维数组，形状为 (时间步数, 特征数)。
            label (int): 样本的标签。
            feature_names (list): 特征名称列表。
            split (object, optional): 分割信息。 Defaults to None.
        """
        self.timestamp = timestamp
        self.features = features
        self.label = label
        self.feature_names = feature_names
        self.split = split  # 添加 split 属性
        self.preprocessed_features = None

    def preprocess(self, all_feature_names, max_time_steps, window_size, scaler):
        """
        预处理时间序列数据。

        这个方法执行以下步骤：
        1. 填充缺失特征
        2. 处理时间步数不足的情况
        3. 填充NaN值
        4. 标准化特征
        5. 创建滑动窗口

        Args:
            all_feature_names (list): 所有可能的特征名称列表。
            max_time_steps (int): 期望的最大时间步数。
            window_size (int): 滑动窗口的大小。
            scaler (StandardScaler): 用于标准化特征的缩放器。
        """
        # 打印原始特征的统计信息
        # print(f"原始特征统计: min={self.features.min():.4f}, max={self.features.max():.4f}, mean={self.features.mean():.4f}")
        
        # 创建一个包含所有特征的新矩阵，初始值为0
        full_features = np.full((self.features.shape[0], len(all_feature_names)), 0.0)
        for feature_name in self.feature_names:
            full_features[:, all_feature_names.index(feature_name)] = self.features[:, self.feature_names.index(feature_name)]
        
        # 打印填充NaN后的统计信息
        # print(f"填充NaN后统计: min={np.nanmin(full_features):.4f}, max={np.nanmax(full_features):.4f}, mean={np.nanmean(full_features):.4f}")

        # 计算需要在开头填充的行数
        padding_rows = max_time_steps - full_features.shape[0]
        if padding_rows > 0:
            # 为每个特征计算平均值（忽略NaN）
            feature_means = np.nanmean(full_features, axis=0)
            
            # 创建填充数组，使用特征平均值
            padding = np.tile(feature_means, (padding_rows, 1))
            
            # 在特征矩阵开头填充
            padded_features = np.vstack((padding, full_features))
            
            # 调整时间戳（每次减少 600 秒，即 10 分钟）
            new_timestamps = np.arange(self.timestamp[0] - 600 * padding_rows, self.timestamp[0], 600)
            self.timestamp = np.concatenate((new_timestamps, self.timestamp))
        else:
            padded_features = full_features

        # 使用SimpleImputer处理剩余的NaN值
        imputer = SimpleImputer(strategy='mean')
        imputed_features = imputer.fit_transform(padded_features)
        
        # 打印Imputer处理后的统计信息
        # print(f"Imputer后统计: min={imputed_features.min():.4f}, max={imputed_features.max():.4f}, mean={imputed_features.mean():.4f}")

        # 使用StandardScaler标准化特征
        scaled_features = scaler.fit_transform(imputed_features)

        # 创建滑动窗口
        # self.preprocessed_features = np.array([scaled_features[i:i+window_size] 
        #                                        for i in range(len(scaled_features) - window_size + 1)])
        # 取最近的window_size个时间步的特征作为样本
        self.preprocessed_features = scaled_features[-window_size:]
        
        # 打印滑动窗口处理后的统计信息
        # print(f"滑动窗口后统计: min={self.preprocessed_features.min():.4f}, max={self.preprocessed_features.max():.4f}, mean={self.preprocessed_features.mean():.4f}")

        # 调整时间戳以匹配预处理后的特征数量
        self.timestamp = self.timestamp[:len(self.preprocessed_features)]

    def get_preprocessed_data(self):
        """
        获取预处理后的数据。

        Returns:
            tuple: (preprocessed_features, label, timestamp)
                - preprocessed_features: 三维数组，形状为 (样本数, 窗口大小, 特征数)
                - label: 整数，表示样本的标签
                - timestamp: 一维数组，表示每个预处理后样本的时间戳
        
        Raises:
            ValueError: 如果数据还未经过预处理
        """
        if self.preprocessed_features is None:
            raise ValueError("数据还未预处理，请先调用 preprocess 方法。")
        return self.preprocessed_features, self.label, self.timestamp

    @classmethod
    def from_dataframe(cls, df, label):
        """
        从DataFrame创建TimeSeriesSample实例。

        Args:
            df (pd.DataFrame): 包含时间序列数据的DataFrame。
                               应包含'timestamp'列和多个特征列。
            label (int): 样本的标签。

        Returns:
            TimeSeriesSample: 新创建的实例。
        """
        timestamps = df['timestamp'].values
        features = df.drop(['timestamp'], axis=1).values
        feature_names = df.columns.drop('timestamp').tolist()
        return cls(features, label, timestamps, feature_names)

    def to_json(self):
        """
        将 TimeSeriesSample 对象转换为 JSON 格式。

        Returns:
            str: JSON格式的字符串表示，包含以下键：
                 - features: 二维列表，原始或预处理后的特征
                 - label: 整数，样本的标签
                 - timestamp: 列表，时间戳
                 - feature_names: 列表，特征名称
        """
        return json.dumps({
            'features': self.preprocessed_features.tolist() if self.preprocessed_features is not None else self.features.tolist(),
            'label': int(self.label),
            'timestamp': self.timestamp.tolist(),
            'feature_names': self.feature_names
        })

    @classmethod
    def from_json(cls, json_str):
        """
        从 JSON 字符串创建 TimeSeriesSample 对象。

        Args:
            json_str (str): JSON格式的字符串，应包含 'features', 'label', 
                            'timestamp', 和 'feature_names' 键。

        Returns:
            TimeSeriesSample: 从JSON创建的对象实例
        """
        data = json.loads(json_str)
        sample = cls(
            timestamp=data['timestamp'],
            features=data['features'],
            label=data['label'],
            feature_names=data.get('feature_names', []),
            split=data.get('split')  # 从 JSON 中获取 split
        )
        if 'preprocessed_features' in data:
            sample.preprocessed_features = np.array(data['preprocessed_features'])
        return sample

    def preprocess_custom(self, all_feature_names, max_time_steps, window_size, feature_stats):
        """
        使用自定义方法预处理样本数据。

        Args:
            all_feature_names (list): 所有特征名称。
            max_time_steps (int): 最大时间步长。
            window_size (int): 时间窗口大小。
            feature_stats (dict): 特征统计信息。
        """
        # 创建一个包含所有特征的新矩阵，初始值为0
        try: 
            self.feature_stats = feature_stats[-1]['data']
        except:
            self.feature_stats = feature_stats
        full_features = np.full((self.features.shape[0], len(all_feature_names)), 0.0)
        for i, name in enumerate(self.feature_names):
            if name in all_feature_names:
                full_features[:, all_feature_names.index(name)] = self.features[:, i]

        # 计算需要在开头填充的行数
        padding_rows = max_time_steps - full_features.shape[0]
        if padding_rows > 0:
            # 为每个特征使用其平均值进行填充
            padding = np.array([[self.feature_stats[name]['mean'] for name in all_feature_names]] * padding_rows)
            padded_features = np.vstack((padding, full_features))
            
            # 调整时间戳（每次减少 600 秒，即 10 分钟）
            new_timestamps = np.arange(self.timestamp[0] - 600 * padding_rows, self.timestamp[0], 600)
            self.timestamp = np.concatenate((new_timestamps, self.timestamp))
        else:
            padded_features = full_features

        # 填充缺失值并归一化
        for i, name in enumerate(all_feature_names):
            column = padded_features[:, i]
            mask = np.isnan(column)
            column[mask] = self.feature_stats[name]['mean']
            
            min_val, max_val = self.feature_stats[name]['min'], self.feature_stats[name]['max']
            if min_val != max_val:
                padded_features[:, i] = (column - min_val) / (max_val - min_val)
            else:
                padded_features[:, i] = 0  # 如果min和max相等，将所有值设为0

        # 只取最后window_size个时间步的特征
        self.preprocessed_features = padded_features[-window_size:]

        # 调整时间戳以匹配预处理后的特征数量
        self.timestamp = self.timestamp[-window_size:]
        self.feature_names = all_feature_names

    def to_dict(self):
        return {
            'timestamp': self.timestamp,
            'features': self.features,
            'label': self.label,
            'feature_names': self.feature_names,
            'split': self.split  # 包含 split 在字典中
        }
