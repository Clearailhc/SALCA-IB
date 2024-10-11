from collections import deque
from datetime import datetime
import json

class ShortTermMemory:
    """短期记忆类,用于存储最近的数据和事件"""

    def __init__(self, max_size=100):
        """
        初始化短期记忆
        :param max_size: 每种类型数据的最大存储量
        """
        self.max_size = max_size
        # 使用deque存储各种类型的最近数据,限制最大长度
        self.recent_predictions = deque(maxlen=max_size)
        self.recent_evaluations = deque(maxlen=max_size)
        self.recent_feedback = deque(maxlen=max_size)
        self.recent_data_characteristics = deque(maxlen=max_size)
        self.recent_external_events = deque(maxlen=max_size)
        self.recent_model_performances = deque(maxlen=max_size)

    def add_prediction(self, prediction):
        """
        添加新的预测结果
        :param prediction: 预测结果
        """
        self.recent_predictions.append({
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction
        })

    def add_evaluation(self, evaluation):
        """
        添加新的评估结果
        :param evaluation: 评估结果
        """
        self.recent_evaluations.append({
            'timestamp': datetime.now().isoformat(),
            'evaluation': evaluation
        })

    def add_feedback(self, feedback):
        """
        添加新的反馈信息
        :param feedback: 反馈信息
        """
        self.recent_feedback.append({
            'timestamp': datetime.now().isoformat(),
            'feedback': feedback
        })

    def add_data_characteristics(self, characteristics):
        """
        添加新的数据特征信息
        :param characteristics: 数据特征信息
        """
        self.recent_data_characteristics.append({
            'timestamp': datetime.now().isoformat(),
            'characteristics': characteristics
        })

    def add_external_event(self, event):
        """
        添加新的外部事件信息
        :param event: 外部事件信息
        """
        self.recent_external_events.append({
            'timestamp': datetime.now().isoformat(),
            'event': event
        })

    def add_model_performance(self, performance):
        """
        添加新的模型性能信息
        :param performance: 模型性能信息
        """
        self.recent_model_performances.append({
            'timestamp': datetime.now().isoformat(),
            'performance': performance
        })

    def get_recent_data(self, data_type, limit=None):
        """
        获取指定类型的最近数据
        :param data_type: 数据类型
        :param limit: 返回数据的数量限制
        :return: 最近的数据列表
        """
        data = getattr(self, f'recent_{data_type}', None)
        if data is None:
            raise ValueError(f"无效的数据类型: {data_type}")
        return list(data)[:limit] if limit else list(data)

class LongTermMemory:
    """长期记忆类,用于存储历史模式和长期趋势"""

    def __init__(self, max_patterns=100):
        """
        初始化长期记忆
        :param max_patterns: 存储的最大模式数量
        """
        self.max_patterns = max_patterns
        self.historical_failure_patterns = []
        self.successful_model_configurations = {}
        self.long_term_performance_trends = []
        self.network_feature_distribution_shifts = []
        self.external_factor_impacts = []
        self.model_performance_history = {}

    def add_failure_pattern(self, pattern):
        """
        添加新的失败模式
        :param pattern: 失败模式信息
        """
        if len(self.historical_failure_patterns) >= self.max_patterns:
            self.historical_failure_patterns.pop(0)
        self.historical_failure_patterns.append({
            'timestamp': datetime.now().isoformat(),
            'pattern': pattern
        })

    def add_model_configuration(self, model_name, configuration):
        """
        添加新的模型配置
        :param model_name: 模型名称
        :param configuration: 模型配置信息
        """
        self.successful_model_configurations[model_name] = {
            'timestamp': datetime.now().isoformat(),
            'configuration': configuration
        }

    def add_performance_trend(self, trend):
        """
        添加新的性能趋势
        :param trend: 性能趋势信息
        """
        self.long_term_performance_trends.append({
            'timestamp': datetime.now().isoformat(),
            'trend': trend
        })

    def add_feature_distribution_shift(self, shift):
        """
        添加新的特征分布变化
        :param shift: 特征分布变化信息
        """
        self.network_feature_distribution_shifts.append({
            'timestamp': datetime.now().isoformat(),
            'shift': shift
        })

    def add_external_factor_impact(self, impact):
        """
        添加新的外部因素影响
        :param impact: 外部因素影响信息
        """
        self.external_factor_impacts.append({
            'timestamp': datetime.now().isoformat(),
            'impact': impact
        })

    def add_model_performance(self, model_name, performance):
        """
        添加新的模型性能
        :param model_name: 模型名称
        :param performance: 模型性能信息
        """
        if model_name not in self.model_performance_history:
            self.model_performance_history[model_name] = []
        self.model_performance_history[model_name].append({
            'timestamp': datetime.now().isoformat(),
            'performance': performance
        })

    def get_historical_data(self, data_type, limit=None):
        """
        获取指定类型的历史数据
        :param data_type: 数据类型
        :param limit: 返回数据的数量限制
        :return: 历史数据列表或字典
        """
        data = getattr(self, data_type, None)
        if data is None:
            raise ValueError(f"无效的数据类型: {data_type}")
        return data[:limit] if limit else data

class MemorySystem:
    """整合短期和长期记忆的记忆系统"""

    def __init__(self, short_term_size=100, long_term_patterns=1000):
        """
        初始化记忆系统
        :param short_term_size: 短期记忆的最大容量
        :param long_term_patterns: 长期记忆中存储的最大模式数量
        """
        self.short_term = ShortTermMemory(max_size=short_term_size)
        self.long_term = LongTermMemory(max_patterns=long_term_patterns)

    def update_short_term(self, memory_type, data):
        """
        更新短期记忆
        :param memory_type: 记忆类型
        :param data: 要添加的数据
        """
        method_name = f'add_{memory_type}'
        if hasattr(self.short_term, method_name):
            getattr(self.short_term, method_name)(data)
        else:
            raise ValueError(f"无效的短期记忆类型: {memory_type}")

    def update_long_term(self, memory_type, data):
        """
        更新长期记忆
        :param memory_type: 记忆类型
        :param data: 要添加的数据
        """
        method_name = f'add_{memory_type}'
        if hasattr(self.long_term, method_name):
            getattr(self.long_term, method_name)(data)
        else:
            raise ValueError(f"无效的长期记忆类型: {memory_type}")

    def update_model_performance(self, model_name, performance):
        """
        更新模型性能到短期和长期记忆
        :param model_name: 模型名称
        :param performance: 性能数据
        """
        # 更新短期记忆
        self.short_term.add_model_performance({model_name: performance})
        
        # 更新长期记忆
        self.long_term.add_model_performance(model_name, performance)

    def get_short_term_memory(self, memory_type, limit=None):
        """
        获取指定类型的最近短期记忆
        :param memory_type: 记忆类型
        :param limit: 返回数据的数量限制
        :return: 最近的短期记忆列表
        """
        return self.short_term.get_recent_data(memory_type, limit)

    def get_long_term_memory(self, memory_type, limit=None):
        """
        获取指定类型的历史长期记忆
        :param memory_type: 记忆类型
        :param limit: 返回数据的数量限制
        :return: 历史长期记忆列表或字典
        """
        return self.long_term.get_historical_data(memory_type, limit)

    def consolidate_memory(self):
        """
        将短期记忆整合到长期记忆中
        这里实现了一个简单的整合逻辑,实际应用中可能需要更复杂的处理
        """
        for feedback in self.short_term.recent_feedback:
            if 'important' in feedback['feedback'].lower():
                self.long_term.add_failure_pattern(feedback['feedback'])

        # 可以添加更多的整合逻辑,如性能趋势、特征分布变化等

    def get_relevant_memory(self, context):
        """
        根据给定的上下文获取相关的记忆
        :param context: 上下文关键词列表
        :return: 包含相关短期和长期记忆的字典
        """
        relevant_memory = {
            'short_term': {},
            'long_term': {}
        }
        
        # 从短期记忆中获取相关信息
        for attr in ['predictions', 'evaluations', 'feedback', 'data_characteristics', 'external_events', 'model_performances']:
            data = self.short_term.get_recent_data(attr)
            relevant_memory['short_term'][attr] = [
                item for item in data if any(keyword in str(item) for keyword in context)
            ]

        # 从长期记忆中获取相关信息
        for attr in ['historical_failure_patterns', 'successful_model_configurations', 'long_term_performance_trends', 'network_feature_distribution_shifts', 'external_factor_impacts', 'model_performance_history']:
            data = self.long_term.get_historical_data(attr)
            relevant_memory['long_term'][attr] = [
                item for item in data if any(keyword in str(item) for keyword in context)
            ]

        return relevant_memory

    def save_to_json(self, file_path):
        """
        将整个记忆系统保存为JSON文件
        :param file_path: 保存的文件路径
        """
        memory_data = {
            'short_term': {
                attr: list(getattr(self.short_term, attr))
                for attr in ['recent_predictions', 'recent_evaluations', 'recent_feedback', 'recent_data_characteristics', 'recent_external_events', 'recent_model_performances']
            },
            'long_term': {
                attr: getattr(self.long_term, attr)
                for attr in ['historical_failure_patterns', 'successful_model_configurations', 'long_term_performance_trends', 'network_feature_distribution_shifts', 'external_factor_impacts', 'model_performance_history']
            }
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_from_json(cls, file_path):
        """
        从JSON文件加载记忆系统
        :param file_path: JSON文件路径
        :return: 加载的MemorySystem实例
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            memory_data = json.load(f)
        
        memory_system = cls()
        
        for attr, data in memory_data['short_term'].items():
            setattr(memory_system.short_term, attr, deque(data, maxlen=memory_system.short_term.max_size))
        
        for attr, data in memory_data['long_term'].items():
            setattr(memory_system.long_term, attr, data)
        
        return memory_system

if __name__ == "__main__":
    # 创建MemorySystem实例
    memory_system = MemorySystem()

    # 添加长期记忆
    historical_patterns = [
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
            'description': '接收功率异常波动引发的信号质量下降',
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
    ]

    for pattern in historical_patterns:
        memory_system.update_long_term('failure_pattern', pattern)

    model_performance = {
        'xgboost': {'accuracy': 0.82, 'precision': 0.80, 'recall': 0.75, 'f1_score': 0.78, 'optimal_window': '2.5小时'},
        'lstm': {'accuracy': 0.85, 'precision': 0.83, 'recall': 0.78, 'f1_score': 0.80, 'optimal_window': '2.5小时'},
        'gru': {'accuracy': 0.84, 'precision': 0.82, 'recall': 0.77, 'f1_score': 0.79, 'optimal_window': '2.5小时'},
        'cnn': {'accuracy': 0.80, 'precision': 0.78, 'recall': 0.73, 'f1_score': 0.75, 'optimal_window': '2小时'},
        'transformer_encoder': {'accuracy': 0.86, 'precision': 0.84, 'recall': 0.79, 'f1_score': 0.81, 'optimal_window': '2.5小时'}
    }

    for model, performance in model_performance.items():
        memory_system.update_model_performance(model, performance)

    # 添加短期记忆
    recent_predictions = [
        {'timestamp': '2023-08-01 10:00:00', 'prediction': 0.85},
        {'timestamp': '2023-08-01 10:10:00', 'prediction': 0.78},
        {'timestamp': '2023-08-01 10:20:00', 'prediction': 0.92}
    ]

    for prediction in recent_predictions:
        memory_system.update_short_term('prediction', prediction)

    recent_feedback = [
        '最近的实验表明，2.5小时的时间窗口在捕捉大多数故障模式方面表现良好。',
        '对于某些快速变化的特征，如接收功率波动，1.5-2小时的窗口可能更合适。',
        '1小时的窗口在捕捉短期波动方面表现出色，特别是对于发送功率相关的问题。',
        '综合考虑各种故障模式，2-2.5小时似乎能够在捕捉短期波动和长期趋势之间取得良好的平衡。'
    ]

    for feedback in recent_feedback:
        memory_system.update_short_term('feedback', feedback)

    # 测试记忆整合
    memory_system.consolidate_memory()

    # 测试获取相关记忆
    context = ['performance', 'f1_score', 'precision', 'recall']
    relevant_memory = memory_system.get_relevant_memory(context)

    print("相关记忆:")
    print(json.dumps(relevant_memory, indent=2, ensure_ascii=False))

    # 保存记忆系统到JSON文件
    memory_system.save_to_json('memory_system.json')

    # 从JSON文件加载记忆系统
    loaded_memory_system = MemorySystem.load_from_json('memory_system.json')

    # 验证加载的记忆
    print("\n加载后的短期记忆:")
    print(json.dumps(loaded_memory_system.short_term.__dict__, indent=2, ensure_ascii=False))
    print("\n加载后的长期记忆:")
    print(json.dumps(loaded_memory_system.long_term.__dict__, indent=2, ensure_ascii=False))