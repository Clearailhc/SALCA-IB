import json
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import numpy as np
import time

class SimpleMemory:
    def __init__(self, name: str, max_size: int = 100):
        self.name = name
        self.max_size = max_size
        self.memory: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def add_data(self, key: str, value: Any) -> None:
        if len(self.memory[key]) >= self.max_size:
            self.memory[key].pop(0)
        self.memory[key].append({
            'timestamp': int(time.time()),  # 使用整数时间戳，精确到秒
            'data': value
        })
        print(f"[{self.name}] {key} 现在包含 {len(self.memory[key])} 条记录")

    def get_data(self, key: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        data = self.memory.get(key, [])
        if limit is not None:
            data = data[-limit:]
        print(f"[{self.name}] 从 {key} 获取了 {len(data)} 条记录")
        return data

    def get_keys(self) -> List[str]:
        return list(self.memory.keys())

class MemorySystem:
    def __init__(self, short_term_size: int = 100, long_term_size: int = 1000):
        self.short_term = SimpleMemory("短期记忆", max_size=short_term_size)
        self.long_term = SimpleMemory("长期记忆", max_size=long_term_size)

    def update_memory(self, memory_type: str, key: str, value: Any) -> None:
        memory = self._get_memory_by_type(memory_type)
        if memory:
            memory.add_data(key, value)
        else:
            print(f"错误: 无效的记忆类型: {memory_type}")

    def get_memory(self, memory_type: str, key: str) -> Any:
        memory = self._get_memory_by_type(memory_type)
        if memory:
            return memory.get_data(key)
        else:
            print(f"错误: 无效的记忆类型: {memory_type}")
            return None

    def get_memory_keys(self, memory_type: str) -> List[str]:
        memory = self._get_memory_by_type(memory_type)
        return memory.get_keys() if memory else []

    def _get_memory_by_type(self, memory_type: str) -> Optional[SimpleMemory]:
        if memory_type == 'short_term':
            return self.short_term
        elif memory_type == 'long_term':
            return self.long_term
        else:
            print(f"错误: 无效的记忆类型: {memory_type}")
            return None

    def consolidate_memory(self) -> None:
        print("开始记忆整合过程")
        # 这里可以添加一些通用的记忆整合逻辑
        print("记忆整合完成")

    def get_relevant_memory(self, context: List[str], memory_type: str = 'both') -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        relevant_memory = {'short_term': {}, 'long_term': {}}
        
        def search_memory(memory: SimpleMemory, memory_type: str) -> None:
            for key in memory.get_keys():
                data = memory.get_data(key)
                relevant_data = [
                    item for item in data 
                    if any(keyword in str(item['data']).lower() for keyword in context)
                ]
                if relevant_data:
                    relevant_memory[memory_type][key] = relevant_data

        if memory_type in ['short_term', 'both']:
            search_memory(self.short_term, 'short_term')
        if memory_type in ['long_term', 'both']:
            search_memory(self.long_term, 'long_term')

        return relevant_memory

    def save_to_json(self, file_path):
        data = {
            'short_term': {},
            'long_term': {}
        }
        for memory_type in ['short_term', 'long_term']:
            memory = self._get_memory_by_type(memory_type)
            if memory:
                for key, subdata in memory.memory.items():
                    data[memory_type][key] = [item['data'] for item in subdata]
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_json(cls, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        memory_system = cls()
        
        for memory_type in ['short_term', 'long_term']:
            if memory_type in data:
                for key, subdata in data[memory_type].items():
                    memory_system.update_memory(memory_type, key, subdata)
        
        return memory_system

    @staticmethod
    def _json_serializable(obj: Any) -> Union[str, TypeError]:
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")

def main():
    memory_system = MemorySystem()

    # 长期记忆
    memory_system.update_memory('long_term', 'feature_list', [
        'ib_device_stat_bias_current_c_0', 'ib_device_stat_bias_current_c_1',
        'ib_device_stat_bias_current_c_2', 'ib_device_stat_bias_current_c_3',
        'ib_device_stat_rx_power_current_c_0', 'ib_device_stat_rx_power_current_c_1',
        'ib_device_stat_rx_power_current_c_2', 'ib_device_stat_rx_power_current_c_3',
        'ib_device_stat_temperature', 'ib_device_stat_tx_power_current_c_0',
        'ib_device_stat_tx_power_current_c_1', 'ib_device_stat_tx_power_current_c_2',
        'ib_device_stat_tx_power_current_c_3', 'ib_device_stat_voltage',
        'ib_device_stat_wavelength'
    ])
    memory_system.update_memory('long_term', 'time_step', 600)  # 10分钟，以秒为单位

    # 短期记忆
    memory_system.update_memory('short_term', 'current_feature_set', [
        'ib_device_stat_bias_current_c_0',
        'ib_device_stat_bias_current_c_1',
        'ib_device_stat_temperature',
        'ib_device_stat_rx_power_current_c_0',
        'ib_device_stat_rx_power_current_c_1'
    ])

    current_time = int(time.time())
    memory_system.update_memory('short_term', 'current_time_range', {
        'start': current_time - 3600,  # 一小时前
        'end': current_time
    })

    memory_system.update_memory('short_term', 'current_model_performance', {
        'XGBoost': {'accuracy': 0.83, 'precision': 0.81, 'recall': 0.76, 'f1_score': 0.79},
        'LSTM': {'accuracy': 0.86, 'precision': 0.84, 'recall': 0.79, 'f1_score': 0.82},
        'Ensemble': {'accuracy': 0.87, 'precision': 0.85, 'recall': 0.80, 'f1_score': 0.83}
    })

    memory_system.update_memory('short_term', 'selected_models', [
        ('LSTM', {'units': 128, 'layers': 3, 'dropout': 0.25}),
        ('CNN', {'filters': 64, 'kernel_size': 3, 'pool_size': 2}),
        ('GRU', {'units': 64, 'layers': 2, 'dropout': 0.2})
    ])

    memory_system.update_memory('short_term', 'ensemble_method', {
        'method': 'averaging',
        'explanation': "考虑到模型的多样性和计算效率，选择了平均集成方法。"
    })

    # ... 其他长期记忆的更新 ...

    # 测试记忆检索
    print("短期记忆类别:", memory_system.get_memory_keys('short_term'))
    print("长期记忆类别:", memory_system.get_memory_keys('long_term'))

    print("\n短期数据记忆:", memory_system.get_memory('short_term', 'current_feature_set'))
    print("\n长期数据记忆:", memory_system.get_memory('long_term', 'feature_list'))
    print("\n模型选择记忆:", memory_system.get_memory('short_term', 'selected_models'))
    print("\n集成方法记忆:", memory_system.get_memory('short_term', 'ensemble_method'))

    # 测试记忆整合
    memory_system.consolidate_memory()

    # 测试相关记忆检索
    context = ['model', 'performance', 'selection', 'ensemble']
    relevant_memory = memory_system.get_relevant_memory(context)
    print("\n相关记忆:", json.dumps(relevant_memory, indent=2, ensure_ascii=False))

    memory_system.save_to_json('src/memory/memory_test.json')


if __name__ == "__main__":
    main()
