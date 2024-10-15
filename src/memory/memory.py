import json
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

class FlexibleMemory:
    def __init__(self, name: str, max_size: int = 100):
        self.name = name
        self.max_size = max_size
        self.memory: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=max_size)))

    def add_data(self, category: str, subcategory: str, data: Any) -> None:
        if category not in self.memory or subcategory not in self.memory[category]:
            print(f"[{self.name}] 创建新的记忆类型: {category}/{subcategory}")
        
        self.memory[category][subcategory].append({
            'timestamp': datetime.now().isoformat(),
            'data': data
        })
        print(f"[{self.name}] {category}/{subcategory} 现在包含 {len(self.memory[category][subcategory])} 条记录")

    def get_data(self, category: str, subcategory: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if category not in self.memory or subcategory not in self.memory[category]:
            print(f"[{self.name}] 警告: 尝试获取不存在的记忆类型 {category}/{subcategory}")
            return []
        
        data = list(self.memory[category][subcategory])
        if limit is not None:
            data = data[-limit:]
        
        print(f"[{self.name}] 从 {category}/{subcategory} 获取了 {len(data)} 条记录")
        return data

    def get_categories(self) -> List[str]:
        return list(self.memory.keys())

    def get_subcategories(self, category: str) -> List[str]:
        return list(self.memory[category].keys()) if category in self.memory else []

class MemorySystem:
    def __init__(self, short_term_size: int = 100, long_term_size: int = 1000):
        self.short_term = FlexibleMemory("短期记忆", max_size=short_term_size)
        self.long_term = FlexibleMemory("长期记忆", max_size=long_term_size)

    def update_memory(self, memory_type: str, category: str, subcategory: str, data: Any) -> None:
        memory = self._get_memory_by_type(memory_type)
        if memory:
            memory.add_data(category, subcategory, data)
        else:
            print(f"错误: 无效的记忆类型: {memory_type}")

    def get_memory(self, memory_type: str, category: str, subcategory: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        memory = self._get_memory_by_type(memory_type)
        return memory.get_data(category, subcategory, limit) if memory else []

    def get_memory_categories(self, memory_type: str) -> List[str]:
        memory = self._get_memory_by_type(memory_type)
        return memory.get_categories() if memory else []

    def get_memory_subcategories(self, memory_type: str, category: str) -> List[str]:
        memory = self._get_memory_by_type(memory_type)
        return memory.get_subcategories(category) if memory else []

    def _get_memory_by_type(self, memory_type: str) -> Optional[FlexibleMemory]:
        if memory_type == 'short_term':
            return self.short_term
        elif memory_type == 'long_term':
            return self.long_term
        else:
            print(f"错误: 无效的记忆类型: {memory_type}")
            return None

    def consolidate_memory(self) -> None:
        print("开始记忆整合过程")
        for category in self.short_term.get_categories():
            for subcategory in self.short_term.get_subcategories(category):
                short_term_data = self.short_term.get_data(category, subcategory)
                for item in short_term_data:
                    if self._is_important(item['data']):
                        self.long_term.add_data(category, subcategory, item['data'])
        print("记忆整合完成")

    def _is_important(self, data: Any) -> bool:
        # 实现重要性判断逻辑
        return True  # 示例：所有数据都被认为是重要的

    def get_relevant_memory(self, context: List[str], memory_type: str = 'both') -> Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]]:
        relevant_memory = {'short_term': {}, 'long_term': {}}
        
        def search_memory(memory: FlexibleMemory, memory_type: str) -> None:
            for category in memory.get_categories():
                relevant_memory[memory_type][category] = {}
                for subcategory in memory.get_subcategories(category):
                    data = memory.get_data(category, subcategory)
                    relevant_data = [
                        item for item in data 
                        if any(keyword in str(item['data']).lower() for keyword in context)
                    ]
                    if relevant_data:
                        relevant_memory[memory_type][category][subcategory] = relevant_data

        if memory_type in ['short_term', 'both']:
            search_memory(self.short_term, 'short_term')
        if memory_type in ['long_term', 'both']:
            search_memory(self.long_term, 'long_term')

        return relevant_memory

    def save_to_json(self, file_path: str) -> None:
        memory_data = {
            'short_term': {category: {subcategory: list(data) for subcategory, data in subcategories.items()}
                           for category, subcategories in self.short_term.memory.items()},
            'long_term': {category: {subcategory: list(data) for subcategory, data in subcategories.items()}
                          for category, subcategories in self.long_term.memory.items()}
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, ensure_ascii=False, indent=2, default=json_serializable)
        
        print(f"记忆系统已保存到 {file_path}")

    @classmethod
    def load_from_json(cls, file_path: str) -> 'MemorySystem':
        with open(file_path, 'r', encoding='utf-8') as f:
            memory_data = json.load(f)
        
        memory_system = cls()
        for term in ['short_term', 'long_term']:
            for category, subcategories in memory_data[term].items():
                for subcategory, data in subcategories.items():
                    for item in data:
                        memory_system.update_memory(term, category, subcategory, item['data'])
        
        print(f"记忆系统已从 {file_path} 加载")
        return memory_system

def json_serializable(obj: Any) -> Union[str, TypeError]:
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

def main():
    memory_system = MemorySystem()

    # 添加数据相关的长期记忆
    best_feature_combinations = [
        {
            'features': ['ib_device_stat_bias_current_c_0', 'ib_device_stat_bias_current_c_1', 'ib_device_stat_bias_current_c_2', 'ib_device_stat_bias_current_c_3', 'ib_device_stat_temperature'],
            'performance': 0.85,
            'description': '高偏置电流导致的设备过热故障模式'
        },
        {
            'features': ['ib_device_stat_rx_power_current_c_0', 'ib_device_stat_rx_power_current_c_1', 'ib_device_stat_rx_power_current_c_2', 'ib_device_stat_rx_power_current_c_3'],
            'performance': 0.82,
            'description': '接收功率异常波动引发的信号质量下降'
        },
        {
            'features': ['ib_device_stat_tx_power_current_c_0', 'ib_device_stat_tx_power_current_c_1', 'ib_device_stat_tx_power_current_c_2', 'ib_device_stat_tx_power_current_c_3'],
            'performance': 0.80,
            'description': '发送功率过高导致的信号干扰问题'
        }
    ]
    for combo in best_feature_combinations:
        memory_system.update_memory('long_term', 'data', 'best_feature_combinations', combo)

    time_step_selections = {
        'normal': 60,
        'high_traffic': 30,
        'low_traffic': 120,
        'failure_prediction': 45
    }
    memory_system.update_memory('long_term', 'data', 'time_step_selections', time_step_selections)

    window_sizes = {
        'bias_current_fault': '2.5小时',
        'rx_power_fluctuation': '1.5-2小时',
        'tx_power_interference': '1-1.5小时',
        'general_fault_prediction': '2-2.5小时'
    }
    memory_system.update_memory('long_term', 'data', 'optimal_window_sizes', window_sizes)

    normalization_methods = {
        'MinMaxScaler': {'performance': 0.83, 'best_for': ['bias_current', 'temperature']},
        'StandardScaler': {'performance': 0.85, 'best_for': ['rx_power', 'tx_power']},
        'RobustScaler': {'performance': 0.81, 'best_for': ['outlier_sensitive_features']}
    }
    memory_system.update_memory('long_term', 'data', 'normalization_methods', normalization_methods)

    # 添加模型相关的长期记忆
    best_model_structures = {
        'XGBoost': {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.1, 'performance': {'accuracy': 0.82, 'f1_score': 0.80}},
        'LSTM': {'units': 64, 'layers': 2, 'dropout': 0.2, 'performance': {'accuracy': 0.85, 'f1_score': 0.83}},
        'GRU': {'units': 32, 'layers': 2, 'dropout': 0.3, 'performance': {'accuracy': 0.84, 'f1_score': 0.82}},
        'CNN': {'filters': 64, 'kernel_size': 3, 'pool_size': 2, 'performance': {'accuracy': 0.80, 'f1_score': 0.78}},
        'Transformer': {'num_heads': 8, 'ff_dim': 32, 'performance': {'accuracy': 0.86, 'f1_score': 0.84}}
    }
    memory_system.update_memory('long_term', 'model', 'best_model_structures', best_model_structures)

    model_ensemble_strategies = [
        {'strategy': 'Voting', 'models': ['XGBoost', 'LSTM', 'GRU'], 'weights': [0.3, 0.4, 0.3], 'performance': 0.87},
        {'strategy': 'Stacking', 'base_models': ['XGBoost', 'LSTM', 'CNN'], 'meta_model': 'LogisticRegression', 'performance': 0.88},
        {'strategy': 'Bagging', 'base_model': 'DecisionTree', 'n_estimators': 10, 'performance': 0.83}
    ]
    memory_system.update_memory('long_term', 'model', 'ensemble_strategies', model_ensemble_strategies)

    # 添加反馈相关的长期记忆
    typical_failure_patterns = [
        {
            'pattern_id': 1,
            'description': '高偏置电流导致的设备过热故障模式',
            'features': ['ib_device_stat_bias_current_c_0', 'ib_device_stat_bias_current_c_1', 'ib_device_stat_bias_current_c_2', 'ib_device_stat_bias_current_c_3', 'ib_device_stat_temperature'],
            'typical_duration': '1-3小时',
            'occurrence_rate': 0.04,
            'recommended_window': '2-3小时'
        },
        {
            'pattern_id': 2,
            'description': '接收功率异常波动引发的信号质量下降',
            'features': ['ib_device_stat_rx_power_current_c_0', 'ib_device_stat_rx_power_current_c_1', 'ib_device_stat_rx_power_current_c_2', 'ib_device_stat_rx_power_current_c_3'],
            'typical_duration': '1-2小时',
            'occurrence_rate': 0.03,
            'recommended_window': '1.5-2小时'
        },
        {
            'pattern_id': 3,
            'description': '发送功率过高导致的信号干扰问题',
            'features': ['ib_device_stat_tx_power_current_c_0', 'ib_device_stat_tx_power_current_c_1', 'ib_device_stat_tx_power_current_c_2', 'ib_device_stat_tx_power_current_c_3'],
            'typical_duration': '30分钟-2小时',
            'occurrence_rate': 0.02,
            'recommended_window': '1-1.5小时'
        }
    ]
    for pattern in typical_failure_patterns:
        memory_system.update_memory('long_term', 'feedback', 'typical_failure_patterns', pattern)

    long_term_model_performance = {
        'XGBoost': {'accuracy': 0.82, 'precision': 0.80, 'recall': 0.75, 'f1_score': 0.78, 'optimal_window': '2.5小时'},
        'LSTM': {'accuracy': 0.85, 'precision': 0.83, 'recall': 0.78, 'f1_score': 0.80, 'optimal_window': '2.5小时'},
        'GRU': {'accuracy': 0.84, 'precision': 0.82, 'recall': 0.77, 'f1_score': 0.79, 'optimal_window': '2.5小时'},
        'CNN': {'accuracy': 0.80, 'precision': 0.78, 'recall': 0.73, 'f1_score': 0.75, 'optimal_window': '2小时'},
        'Transformer': {'accuracy': 0.86, 'precision': 0.84, 'recall': 0.79, 'f1_score': 0.81, 'optimal_window': '2.5小时'}
    }
    memory_system.update_memory('long_term', 'feedback', 'long_term_model_performance', long_term_model_performance)

    # 添加数据相关的短期记忆
    current_feature_set = [
        'ib_device_stat_bias_current_c_0',
        'ib_device_stat_bias_current_c_1',
        'ib_device_stat_temperature',
        'ib_device_stat_rx_power_current_c_0',
        'ib_device_stat_rx_power_current_c_1'
    ]
    memory_system.update_memory('short_term', 'data', 'current_feature_set', current_feature_set)

    current_time_step = 45
    memory_system.update_memory('short_term', 'data', 'current_time_step', current_time_step)

    current_window_size = '2小时'
    memory_system.update_memory('short_term', 'data', 'current_window_size', current_window_size)

    current_scaler = 'StandardScaler'
    memory_system.update_memory('short_term', 'data', 'current_scaler', current_scaler)

    # 添加模型相关的短期记忆
    current_model_performance = {
        'XGBoost': {'accuracy': 0.83, 'precision': 0.81, 'recall': 0.76, 'f1_score': 0.79},
        'LSTM': {'accuracy': 0.86, 'precision': 0.84, 'recall': 0.79, 'f1_score': 0.82},
        'Ensemble': {'accuracy': 0.87, 'precision': 0.85, 'recall': 0.80, 'f1_score': 0.83}
    }
    memory_system.update_memory('short_term', 'model', 'current_model_performance', current_model_performance)

    current_hyperparameters = {
        'XGBoost': {'max_depth': 7, 'n_estimators': 120, 'learning_rate': 0.08},
        'LSTM': {'units': 128, 'layers': 3, 'dropout': 0.25},
        'Ensemble': {'strategy': 'Voting', 'weights': [0.4, 0.6]}
    }
    memory_system.update_memory('short_term', 'model', 'current_hyperparameters', current_hyperparameters)

    # 添加反馈相关的短期记忆
    recent_predictions = [
        {'timestamp': '2023-05-01 10:00:00', 'model': 'XGBoost', 'prediction': 0.8, 'actual': 1, 'features': ['ib_device_stat_bias_current_c_0', 'ib_device_stat_temperature']},
        {'timestamp': '2023-05-01 11:00:00', 'model': 'LSTM', 'prediction': 0.7, 'actual': 0, 'features': ['ib_device_stat_rx_power_current_c_0', 'ib_device_stat_rx_power_current_c_1']},
        {'timestamp': '2023-05-01 12:00:00', 'model': 'Ensemble', 'prediction': 0.9, 'actual': 1, 'features': ['ib_device_stat_bias_current_c_0', 'ib_device_stat_rx_power_current_c_0']}
    ]
    for prediction in recent_predictions:
        memory_system.update_memory('short_term', 'feedback', 'recent_predictions', prediction)

    recent_model_evaluations = [
        {'timestamp': '2023-05-01', 'model': 'XGBoost', 'accuracy': 0.84, 'f1_score': 0.81, 'training_time': '120s'},
        {'timestamp': '2023-05-01', 'model': 'LSTM', 'accuracy': 0.86, 'f1_score': 0.83, 'training_time': '300s'},
        {'timestamp': '2023-05-01', 'model': 'Ensemble', 'accuracy': 0.88, 'f1_score': 0.85, 'training_time': '450s'}
    ]
    for evaluation in recent_model_evaluations:
        memory_system.update_memory('short_term', 'feedback', 'recent_model_evaluations', evaluation)

    recent_failure_patterns = [
        {
            'timestamp': '2023-05-01 09:30:00',
            'pattern_id': 1,
            'description': '高偏置电流导致的设备过热',
            'detected_features': ['ib_device_stat_bias_current_c_0', 'ib_device_stat_temperature'],
            'duration': '2小时15分钟'
        },
        {
            'timestamp': '2023-05-01 14:45:00',
            'pattern_id': 2,
            'description': '接收功率异常波动',
            'detected_features': ['ib_device_stat_rx_power_current_c_0', 'ib_device_stat_rx_power_current_c_1'],
            'duration': '1小时30分钟'
        }
    ]
    for pattern in recent_failure_patterns:
        memory_system.update_memory('short_term', 'feedback', 'recent_failure_patterns', pattern)

    # 测试记忆检索
    print("短期记忆类别:", memory_system.get_memory_categories('short_term'))
    print("长期记忆类别:", memory_system.get_memory_categories('long_term'))

    print("\n短期数据记忆:", memory_system.get_memory('short_term', 'data', 'current_feature_set'))
    print("\n长期模型记忆:", memory_system.get_memory('long_term', 'model', 'best_model_structures'))

    # 测试记忆整合
    memory_system.consolidate_memory()

    # 测试相关记忆检索
    context = ['performance', 'feature', 'model', 'failure']
    relevant_memory = memory_system.get_relevant_memory(context)
    print("\n相关记忆:", json.dumps(relevant_memory, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()