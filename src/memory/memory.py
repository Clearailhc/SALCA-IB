import json
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

class FlexibleMemory:
    def __init__(self, name: str, max_size: int = 100):
        self.name = name
        self.max_size = max_size
        self.memory: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_size))

    def add_data(self, category: str, data: Any) -> None:
        if category not in self.memory:
            print(f"[{self.name}] 创建新的记忆类型: {category}")
        
        self.memory[category].append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data': data
        })
        print(f"[{self.name}] {category} 现在包含 {len(self.memory[category])} 条记录")

    def get_data(self, category: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if category not in self.memory:
            print(f"[{self.name}] 警告: 尝试获取不存在的记忆类型 {category}")
            return []
        
        data = list(self.memory[category])
        if limit is not None:
            data = data[-limit:]
        
        print(f"[{self.name}] 从 {category} 获取了 {len(data)} 条记录")
        return data

    def get_categories(self) -> List[str]:
        return list(self.memory.keys())

class MemorySystem:
    def __init__(self, short_term_size: int = 100, long_term_size: int = 1000):
        self.short_term = FlexibleMemory("短期记忆", max_size=short_term_size)
        self.long_term = FlexibleMemory("长期记忆", max_size=long_term_size)

    def update_memory(self, memory_type: str, category: str, data: Any) -> None:
        memory = self._get_memory_by_type(memory_type)
        if memory:
            memory.add_data(category, data)
        else:
            print(f"错误: 无效的记忆类型: {memory_type}")

    def get_memory(self, memory_type: str, category: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        memory = self._get_memory_by_type(memory_type)
        return memory.get_data(category, limit) if memory else []

    def get_memory_categories(self, memory_type: str) -> List[str]:
        memory = self._get_memory_by_type(memory_type)
        return memory.get_categories() if memory else []

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
            short_term_data = self.short_term.get_data(category)
            for item in short_term_data:
                if 'important' in str(item['data']).lower():
                    self.long_term.add_data(category, item['data'])
        print("记忆整合完成")

    def get_relevant_memory(self, context: List[str], memory_type: str = 'both') -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        relevant_memory = {'short_term': {}, 'long_term': {}}
        
        def search_memory(memory: FlexibleMemory, memory_type: str) -> None:
            for category in memory.get_categories():
                data = memory.get_data(category)
                relevant_data = [
                    item for item in data 
                    if any(keyword in str(item['data']).lower() for keyword in context)
                ]
                if relevant_data:
                    relevant_memory[memory_type][category] = relevant_data

        if memory_type in ['short_term', 'both']:
            search_memory(self.short_term, 'short_term')
        if memory_type in ['long_term', 'both']:
            search_memory(self.long_term, 'long_term')

        return relevant_memory

    def save_to_json(self, file_path: str) -> None:
        memory_data = {
            'short_term': {k: list(v) for k, v in self.short_term.memory.items()},
            'long_term': {k: list(v) for k, v in self.long_term.memory.items()}
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
            for category, data in memory_data[term].items():
                for item in data:
                    getattr(memory_system, term).add_data(category, item['data'])
        
        print(f"记忆系统已从 {file_path} 加载")
        return memory_system

def json_serializable(obj: Any) -> Union[str, TypeError]:
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

def main():
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
        memory_system.update_memory('long_term', 'failure_patterns', pattern)

    model_performance = {
        'xgboost': {'accuracy': 0.82, 'precision': 0.80, 'recall': 0.75, 'f1_score': 0.78, 'optimal_window': '2.5小时'},
        'lstm': {'accuracy': 0.85, 'precision': 0.83, 'recall': 0.78, 'f1_score': 0.80, 'optimal_window': '2.5小时'},
        'gru': {'accuracy': 0.84, 'precision': 0.82, 'recall': 0.77, 'f1_score': 0.79, 'optimal_window': '2.5小时'},
        'cnn': {'accuracy': 0.80, 'precision': 0.78, 'recall': 0.73, 'f1_score': 0.75, 'optimal_window': '2小时'},
        'transformer_encoder': {'accuracy': 0.86, 'precision': 0.84, 'recall': 0.79, 'f1_score': 0.81, 'optimal_window': '2.5小时'}
    }

    for model, performance in model_performance.items():
        memory_system.update_memory('long_term', 'model_performance', {model: performance})

    # 添加短期记忆
    recent_predictions = [
        {
            'model': 'XGBoost',
            'prediction': 0.85,
            'actual': 1,
            'features': ['ib_device_stat_bias_current_c_0', 'ib_device_stat_temperature'],
            'performance': {'accuracy': 0.82, 'precision': 0.80, 'recall': 0.75, 'f1_score': 0.78}
        },
        {
            'model': 'LSTM',
            'prediction': 0.78,
            'actual': 0,
            'features': ['ib_device_stat_rx_power_current_c_0', 'ib_device_stat_rx_power_current_c_1'],
            'performance': {'accuracy': 0.85, 'precision': 0.83, 'recall': 0.78, 'f1_score': 0.80}
        },
        {
            'model': 'Transformer',
            'prediction': 0.92,
            'actual': 1,
            'features': ['ib_device_stat_tx_power_current_c_0', 'ib_device_stat_tx_power_current_c_1'],
            'performance': {'accuracy': 0.86, 'precision': 0.84, 'recall': 0.79, 'f1_score': 0.81}
        }
    ]

    for prediction in recent_predictions:
        memory_system.update_memory('short_term', 'predictions', prediction)

    recent_feedback = [
        '最近的实验表明，2.5小时的时间窗口在捕捉大多数故障模式方面表现良好。',
        '对于某些快速变化的特征，如接收功率波动，1.5-2小时的窗口可能更合适。',
        '1小时的窗口在捕捉短期波动方面表现出色，特别是对于发送功率相关的问题。',
        '综合考虑各种故障模式，2-2.5小时似乎能够在捕捉短期波动和长期趋势之间取得良好的平衡。'
    ]

    for feedback in recent_feedback:
        memory_system.update_memory('short_term', 'feedback', feedback)

    # 测试记忆检索
    print("短期记忆类别:", memory_system.get_memory_categories('short_term'))
    print("长期记忆类别:", memory_system.get_memory_categories('long_term'))

    print("\n短期预测:", memory_system.get_memory('short_term', 'predictions'))
    print("\n长期故障模式:", memory_system.get_memory('long_term', 'failure_patterns'))

    # 测试记忆整合
    memory_system.consolidate_memory()

    # 测试相关记忆检索
    context = ['performance', 'window', 'model']
    relevant_memory = memory_system.get_relevant_memory(context)
    print("\n相关记忆:", json.dumps(relevant_memory, indent=2, ensure_ascii=False))

    # 保存和加载测试
    memory_system.save_to_json('./src/memory/memory_test.json')
    loaded_memory = MemorySystem.load_from_json('./src/memory/memory_test.json')

    print("\n加载后的短期记忆类别:", loaded_memory.get_memory_categories('short_term'))
    print("加载后的长期记忆类别:", loaded_memory.get_memory_categories('long_term'))

if __name__ == "__main__":
    main()