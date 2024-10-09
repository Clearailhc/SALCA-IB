import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import json
from datetime import timedelta

class TimeSeriesSample:
    def __init__(self, features, label, timestamp, feature_names):
        self.features = features
        self.label = label
        self.timestamp = timestamp
        self.feature_names = feature_names
        self.preprocessed_features = None

    def preprocess(self, all_feature_names, max_time_steps, window_size, scaler):
        print(f"原始特征统计: min={self.features.min():.4f}, max={self.features.max():.4f}, mean={self.features.mean():.4f}")
        
        # 创建一个新的特征矩阵，包含所有特征
        full_features = np.full((self.features.shape[0], len(all_feature_names)), np.nan)
        for i, feature_name in enumerate(self.feature_names):
            full_features[:, all_feature_names.index(feature_name)] = self.features[:, i]
        
        print(f"填充NaN后统计: min={np.nanmin(full_features):.4f}, max={np.nanmax(full_features):.4f}, mean={np.nanmean(full_features):.4f}")

        # 计算需要在开头填充的行数
        padding_rows = max_time_steps - full_features.shape[0]
        if padding_rows > 0:
            # 为每个特征计算平均值（忽略NaN）
            feature_means = np.nanmean(full_features, axis=0)
            
            # 创建填充数组
            padding = np.tile(feature_means, (padding_rows, 1))
            
            # 在开头填充特征
            padded_features = np.vstack((padding, full_features))
            
            # 调整时间戳（每次减少 600 秒，即 10 分钟）
            new_timestamps = np.arange(self.timestamp[0] - 600 * padding_rows, self.timestamp[0], 600)
            self.timestamp = np.concatenate((new_timestamps, self.timestamp))
        else:
            padded_features = full_features

        # print(f"时间步填充后统计: min={np.nanmin(padded_features):.4f}, max={np.nanmax(padded_features):.4f}, mean={np.nanmean(padded_features):.4f}")

        # 使用SimpleImputer处理剩余的NaN值
        imputer = SimpleImputer(strategy='mean')
        imputed_features = imputer.fit_transform(padded_features)
        # 
        print(f"Imputer后统计: min={imputed_features.min():.4f}, max={imputed_features.max():.4f}, mean={imputed_features.mean():.4f}")

        # 标准化特征
        scaled_features = scaler.fit_transform(imputed_features)
        
        # print(f"标准化后统计: min={scaled_features.min():.4f}, max={scaled_features.max():.4f}, mean={scaled_features.mean():.4f}")

        # 创建滑动窗口
        self.preprocessed_features = np.array([scaled_features[i:i+window_size] 
                                               for i in range(len(scaled_features) - window_size + 1)])
        
        print(f"滑动窗口后统计: min={self.preprocessed_features.min():.4f}, max={self.preprocessed_features.max():.4f}, mean={self.preprocessed_features.mean():.4f}")

        # 调整时间戳以匹配预处理后的特征
        self.timestamp = self.timestamp[:len(self.preprocessed_features)]

    def get_preprocessed_data(self):
        """
        获取预处理后的数据。

        Returns:
            tuple: (preprocessed_features, label, timestamp)
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
        """
        data = json.loads(json_str)
        sample = cls(
            np.array(data['features']),
            data['label'],
            np.array(data['timestamp']),
            data['feature_names']
        )
        if 'preprocessed_features' in data:
            sample.preprocessed_features = np.array(data['preprocessed_features'])
        return sample
