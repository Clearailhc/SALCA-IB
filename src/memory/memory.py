import json
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import numpy as np

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
        # 初始化长期记忆的 'model_performance' 类别
        self.update_memory('long_term', 'model_performance', 'trends', {})
        # 初始化长期记忆的 'hyperparameters' 类别
        self.update_memory('long_term', 'hyperparameters', 'optimal_ranges', {})

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
        self._update_model_performance_trends()
        self._update_best_feature_combinations()
        self._update_optimal_hyperparameter_ranges()
        self._update_failure_patterns()
        self._update_optimal_time_window()
        self._update_ensemble_strategy_effectiveness()
        self._update_data_distribution_changes()
        print("记忆整合完成")

    def _update_model_performance_trends(self):
        short_term_performance = self.get_memory('short_term', 'model', 'current_model_performance')
        if not short_term_performance or not isinstance(short_term_performance[-1]['data'], dict):
            print("警告: 无效的短期性能数据")
            return

        trends = self.get_memory('long_term', 'model_performance', 'trends')
        if not isinstance(trends, dict):
            trends = {}

        for model, performance in short_term_performance[-1]['data'].items():
            if not isinstance(model, str) or not isinstance(performance, dict):
                print(f"警告: 无效的模型性能数据: {model}, {performance}")
                continue
            if model not in trends:
                trends[model] = {'accuracy_trend': [], 'f1_trend': []}
            trends[model]['accuracy_trend'].append(performance.get('accuracy', 0))
            trends[model]['f1_trend'].append(performance.get('f1_score', 0))

            # 只保留最近的5个数据点
            trends[model]['accuracy_trend'] = trends[model]['accuracy_trend'][-5:]
            trends[model]['f1_trend'] = trends[model]['f1_trend'][-5:]

        self.update_memory('long_term', 'model_performance', 'trends', trends)

    def _update_best_feature_combinations(self):
        short_term_features = self.get_memory('short_term', 'data', 'current_feature_set')
        short_term_performance = self.get_memory('short_term', 'model', 'current_model_performance')
        if not short_term_features or not short_term_performance:
            return

        best_combinations = self.get_memory('long_term', 'feature_selection', 'best_combinations')
        if not isinstance(best_combinations, list):
            best_combinations = []

        current_features = short_term_features[-1]['data']
        if not isinstance(current_features, list):
            print(f"警告: 无效的特征集: {current_features}")
            return

        current_performance = max(short_term_performance[-1]['data'].values(), key=lambda x: x.get('accuracy', 0))

        new_combination = {
            'features': current_features,
            'performance': current_performance.get('accuracy', 0),
            'model': max(short_term_performance[-1]['data'], key=lambda k: short_term_performance[-1]['data'][k].get('accuracy', 0))
        }

        # 检查并修复现有组合
        valid_combinations = []
        for item in best_combinations:
            if not isinstance(item, dict):
                print(f"警告: 发现非字典项: {item}")
                continue
            if 'features' not in item:
                print(f"警告: 发现没有 'features' 键的项: {item}")
                continue
            if 'performance' not in item:
                print(f"警告: 发现没有 'performance' 键的项: {item}")
                item['performance'] = 0  # 设置默认值
            valid_combinations.append(item)

        # 检查是否已存在相同的特征组合
        existing_combination = next((item for item in valid_combinations if item['features'] == current_features), None)
        if existing_combination:
            existing_combination.update(new_combination)
        else:
            valid_combinations.append(new_combination)

        # 对组合进行排序并只保留前5个
        try:
            valid_combinations.sort(key=lambda x: x['performance'], reverse=True)
        except KeyError as e:
            print(f"排序时出错: {e}")
            print("当前的 valid_combinations:", valid_combinations)
        valid_combinations = valid_combinations[:5]

        self.update_memory('long_term', 'feature_selection', 'best_combinations', valid_combinations)

    def _update_optimal_hyperparameter_ranges(self):
        short_term_models = self.get_memory('short_term', 'model_selection', 'selected_models')
        if not short_term_models:
            return

        optimal_ranges = self.get_memory('long_term', 'hyperparameters', 'optimal_ranges')
        if not isinstance(optimal_ranges, dict):
            optimal_ranges = {}

        for model_data in short_term_models[-1]['data']:
            if isinstance(model_data, tuple) and len(model_data) == 2:
                model, params = model_data
            else:
                print(f"警告: 无效的模型数据格式: {model_data}")
                continue

            if not isinstance(model, str) or not isinstance(params, dict):
                print(f"警告: 模型名称应为字符串，参数应为字典: {model}, {params}")
                continue

            if model not in optimal_ranges:
                optimal_ranges[model] = {}

            for param, value in params.items():
                if param not in optimal_ranges[model]:
                    optimal_ranges[model][param] = [value, value]
                else:
                    optimal_ranges[model][param][0] = min(optimal_ranges[model][param][0], value)
                    optimal_ranges[model][param][1] = max(optimal_ranges[model][param][1], value)

        self.update_memory('long_term', 'hyperparameters', 'optimal_ranges', optimal_ranges)

    def _update_failure_patterns(self):
        short_term_failures = self.get_memory('short_term', 'feedback', 'recent_failure_patterns')
        if not short_term_failures:
            return

        failure_patterns = self.get_memory('long_term', 'failure_patterns', 'feature_correlations')
        if not isinstance(failure_patterns, list):
            failure_patterns = []

        for failure in short_term_failures[-1]['data']:
            if not isinstance(failure, dict) or 'description' not in failure or 'detected_features' not in failure:
                print(f"警告: 无效的故障模式数据: {failure}")
                continue
            existing_pattern = next((p for p in failure_patterns if p.get('pattern') == failure['description']), None)
            if existing_pattern:
                existing_pattern['key_features'] = failure['detected_features']
                existing_pattern['correlation_strength'] = 0.8  # 这里可以实现更复杂的相关性计算
            else:
                failure_patterns.append({
                    'pattern': failure['description'],
                    'key_features': failure['detected_features'],
                    'correlation_strength': 0.8
                })

        self.update_memory('long_term', 'failure_patterns', 'feature_correlations', failure_patterns)

    def _update_optimal_time_window(self):
        short_term_window = self.get_memory('short_term', 'data', 'time_window')
        short_term_performance = self.get_memory('short_term', 'model', 'current_model_performance')
        if not short_term_window or not short_term_performance:
            return

        optimal_window = self.get_memory('long_term', 'data_preprocessing', 'optimal_time_window')
        if not isinstance(optimal_window, dict):
            optimal_window = {'size': 0, 'performance_improvement': 0}

        current_window = short_term_window[-1]['data']
        if not isinstance(current_window, (int, float)):
            print(f"警告: 无效的时间窗口大小: {current_window}")
            return

        current_performance = max(short_term_performance[-1]['data'].values(), key=lambda x: x.get('accuracy', 0)).get('accuracy', 0)

        if current_performance > optimal_window['performance_improvement']:
            optimal_window = {
                'size': current_window,
                'performance_improvement': current_performance,
                'explanation': f"{current_window}秒的时间窗口表现最佳，模型性能达到{current_performance:.2f}"
            }

        self.update_memory('long_term', 'data_preprocessing', 'optimal_time_window', optimal_window)

    def _update_ensemble_strategy_effectiveness(self):
        short_term_ensemble = self.get_memory('short_term', 'ensemble_method', 'method')
        short_term_performance = self.get_memory('short_term', 'model', 'current_model_performance')
        if not short_term_ensemble or not short_term_performance:
            return

        effectiveness = self.get_memory('long_term', 'ensemble_strategies', 'effectiveness')
        if not isinstance(effectiveness, list):
            effectiveness = []

        ensemble_method = short_term_ensemble[-1]['data']
        if not isinstance(ensemble_method, str):
            print(f"警告: 无效的集成方法: {ensemble_method}")
            return

        ensemble_performance = short_term_performance[-1]['data'].get('Ensemble', {}).get('accuracy', 0)
        best_single_model_performance = max(
            (v.get('accuracy', 0) for k, v in short_term_performance[-1]['data'].items() if k != 'Ensemble'),
            default=0
        )

        performance_gain = ensemble_performance - best_single_model_performance

        existing_strategy = next((s for s in effectiveness if s.get('strategy') == ensemble_method), None)
        if existing_strategy:
            existing_strategy['average_performance_gain'] = (existing_strategy['average_performance_gain'] + performance_gain) / 2
        else:
            effectiveness.append({
                'strategy': ensemble_method,
                'average_performance_gain': performance_gain,
                'best_model_combination': list(short_term_performance[-1]['data'].keys())
            })

        self.update_memory('long_term', 'ensemble_strategies', 'effectiveness', effectiveness)

    def _update_data_distribution_changes(self):
        short_term_data = self.get_memory('short_term', 'data', 'current_feature_set')
        if not short_term_data:
            return

        distribution_changes = self.get_memory('long_term', 'data_distribution', 'changes')
        if not distribution_changes:
            distribution_changes = []

        current_features = short_term_data[-1]['data']
        for feature in current_features:
            feature_data = self.get_memory('short_term', 'data', feature)
            if not feature_data:
                continue

            current_mean = np.mean(feature_data[-1]['data'])
            current_range = [np.min(feature_data[-1]['data']), np.max(feature_data[-1]['data'])]

            existing_change = next((c for c in distribution_changes if c['feature'] == feature), None)
            if existing_change:
                if abs(current_mean - existing_change['initial_mean']) > 0.1 * existing_change['initial_mean']:
                    existing_change['current_mean'] = current_mean
                    existing_change['change_date'] = datetime.now().strftime('%Y-%m-%d')
                    existing_change['potential_cause'] = '数据分布发生显著变化，需进一步调查'
            else:
                distribution_changes.append({
                    'feature': feature,
                    'initial_mean': current_mean,
                    'current_mean': current_mean,
                    'initial_range': current_range,
                    'current_range': current_range,
                    'change_date': datetime.now().strftime('%Y-%m-%d'),
                    'potential_cause': '新增特征监控'
                })

        self.update_memory('long_term', 'data_distribution', 'changes', distribution_changes)

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

    def save_to_json(self, file_path):
        data = {
            'short_term': {},
            'long_term': {}
        }
        for memory_type in ['short_term', 'long_term']:
            memory = self._get_memory_by_type(memory_type)
            if memory:
                for category, subcategories in memory.memory.items():
                    data[memory_type][category] = {}
                    for subcategory, subdata in subcategories.items():
                        data[memory_type][category][subcategory] = [item['data'] for item in subdata]
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_json(cls, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        memory_system = cls()
        
        for memory_type in ['short_term', 'long_term']:
            if memory_type in data:
                for category, subcategories in data[memory_type].items():
                    if isinstance(subcategories, dict):
                        for subcategory, subdata in subcategories.items():
                            memory_system.update_memory(memory_type, category, subcategory, subdata)
                    elif isinstance(subcategories, list):
                        # 如果subcategories是一个列表，我们假设它是一个数据列表
                        memory_system.update_memory(memory_type, category, 'data', subcategories)
                    else:
                        print(f"警告: 无法处理的数据类型 {type(subcategories)} 在 {memory_type}/{category}")
        
        return memory_system

    @staticmethod
    def _json_serializable(obj: Any) -> Union[str, TypeError]:
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")

def main():
    memory_system = MemorySystem()

    # 添加数据相关的长期记忆
    feature_list = [
        'ib_device_stat_bias_current_c_0', 'ib_device_stat_bias_current_c_1',
        'ib_device_stat_bias_current_c_2', 'ib_device_stat_bias_current_c_3',
        'ib_device_stat_rx_power_current_c_0', 'ib_device_stat_rx_power_current_c_1',
        'ib_device_stat_rx_power_current_c_2', 'ib_device_stat_rx_power_current_c_3',
        'ib_device_stat_temperature', 'ib_device_stat_tx_power_current_c_0',
        'ib_device_stat_tx_power_current_c_1', 'ib_device_stat_tx_power_current_c_2',
        'ib_device_stat_tx_power_current_c_3', 'ib_device_stat_voltage',
        'ib_device_stat_wavelength'
    ]
    memory_system.update_memory('long_term', 'data', 'feature_list', feature_list)

    time_step = 600  # 10分钟
    memory_system.update_memory('long_term', 'data', 'time_step', time_step)

    # 添加数据相关的短期记忆
    current_feature_set = [
        'ib_device_stat_bias_current_c_0',
        'ib_device_stat_bias_current_c_1',
        'ib_device_stat_temperature',
        'ib_device_stat_rx_power_current_c_0',
        'ib_device_stat_rx_power_current_c_1'
    ]
    memory_system.update_memory('short_term', 'data', 'current_feature_set', current_feature_set)

    current_time_range = {
        'start': 1719488400,  # 开始时间戳
        'end': 1719505800     # 结束时间戳
    }
    memory_system.update_memory('short_term', 'data', 'current_time_range', current_time_range)

    # 添加模型相关的短期记忆
    current_model_performance = {
        'XGBoost': {'accuracy': 0.83, 'precision': 0.81, 'recall': 0.76, 'f1_score': 0.79},
        'LSTM': {'accuracy': 0.86, 'precision': 0.84, 'recall': 0.79, 'f1_score': 0.82},
        'Ensemble': {'accuracy': 0.87, 'precision': 0.85, 'recall': 0.80, 'f1_score': 0.83}
    }
    memory_system.update_memory('short_term', 'model', 'current_model_performance', current_model_performance)

    # 添加模型选择和超参数相关的记忆
    selected_models_with_params = [
        ('LSTM', {'units': 128, 'layers': 3, 'dropout': 0.25}),
        ('CNN', {'filters': 64, 'kernel_size': 3, 'pool_size': 2}),
        ('GRU', {'units': 64, 'layers': 2, 'dropout': 0.2})
    ]
    memory_system.update_memory('short_term', 'model_selection', 'selected_models', selected_models_with_params)
    memory_system.update_memory('short_term', 'model_selection', 'explanation', "根据当前数据特征和历史性能，选择了LSTM、CNN和GRU模型。")

    # 添加集成方法相关的记忆
    memory_system.update_memory('short_term', 'ensemble_method', 'method', 'averaging')
    memory_system.update_memory('short_term', 'ensemble_method', 'explanation', "考虑到模型的多样性和计算效率，选择了平均集成方法。")

    # 添加反馈相关的短期记忆
    recent_predictions = [
        {'timestamp': 1719505800, 'model': 'XGBoost', 'prediction': 0.8, 'actual': 1},
        {'timestamp': 1719505200, 'model': 'LSTM', 'prediction': 0.7, 'actual': 0},
        {'timestamp': 1719504600, 'model': 'Ensemble', 'prediction': 0.9, 'actual': 1}
    ]
    for prediction in recent_predictions:
        memory_system.update_memory('short_term', 'feedback', 'recent_predictions', prediction)

    # 添加新的长期记忆示例
    
    # 1. 模型性能趋势
    model_performance_trends = {
        'LSTM': {'accuracy_trend': [0.82, 0.84, 0.86], 'f1_trend': [0.80, 0.82, 0.84]},
        'CNN': {'accuracy_trend': [0.81, 0.83, 0.85], 'f1_trend': [0.79, 0.81, 0.83]},
        'GRU': {'accuracy_trend': [0.83, 0.85, 0.87], 'f1_trend': [0.81, 0.83, 0.85]},
        'Ensemble': {'accuracy_trend': [0.85, 0.87, 0.89], 'f1_trend': [0.83, 0.85, 0.87]}
    }
    memory_system.update_memory('long_term', 'model_performance', 'trends', model_performance_trends)

    # 2. 最佳特征组合
    best_feature_combinations = [
        {
            'features': ['ib_device_stat_bias_current_c_0', 'ib_device_stat_temperature', 'ib_device_stat_rx_power_current_c_0'],
            'performance': 0.88,
            'model': 'LSTM'
        },
        {
            'features': ['ib_device_stat_bias_current_c_1', 'ib_device_stat_tx_power_current_c_1', 'ib_device_stat_voltage'],
            'performance': 0.87,
            'model': 'CNN'
        }
    ]
    memory_system.update_memory('long_term', 'feature_selection', 'best_combinations', best_feature_combinations)

    # 3. 最优超参数范围
    optimal_hyperparameter_ranges = {
        'LSTM': {'units': [64, 128], 'layers': [2, 3], 'dropout': [0.2, 0.3]},
        'CNN': {'filters': [32, 64], 'kernel_size': [3, 5], 'pool_size': [2]},
        'GRU': {'units': [32, 64], 'layers': [2, 3], 'dropout': [0.1, 0.2]}
    }
    memory_system.update_memory('long_term', 'hyperparameters', 'optimal_ranges', optimal_hyperparameter_ranges)

    # 4. 故障模式与特征关联
    failure_patterns = [
        {
            'pattern': '高偏置电流导致的设备过热',
            'key_features': ['ib_device_stat_bias_current_c_0', 'ib_device_stat_temperature'],
            'correlation_strength': 0.85
        },
        {
            'pattern': '接收功率异常波动',
            'key_features': ['ib_device_stat_rx_power_current_c_0', 'ib_device_stat_rx_power_current_c_1'],
            'correlation_strength': 0.78
        }
    ]
    memory_system.update_memory('long_term', 'failure_patterns', 'feature_correlations', failure_patterns)

    # 5. 最佳时间窗口大小
    optimal_time_window = {
        'size': 3600,  # 秒
        'performance_improvement': 0.05,
        'explanation': "1小时的时间窗口在多次实验中表现最佳，平均提升模型性能5个百分点"
    }
    memory_system.update_memory('long_term', 'data_preprocessing', 'optimal_time_window', optimal_time_window)

    # 6. 模型集成策略效果
    ensemble_strategies_effectiveness = [
        {
            'strategy': 'averaging',
            'average_performance_gain': 0.03,
            'best_model_combination': ['LSTM', 'GRU', 'CNN']
        },
        {
            'strategy': 'weighted_voting',
            'average_performance_gain': 0.04,
            'best_weights': {'LSTM': 0.4, 'GRU': 0.3, 'CNN': 0.3}
        }
    ]
    memory_system.update_memory('long_term', 'ensemble_strategies', 'effectiveness', ensemble_strategies_effectiveness)

    # 7. 数据分布变化
    data_distribution_changes = [
        {
            'feature': 'ib_device_stat_temperature',
            'initial_mean': 40.5,
            'current_mean': 42.3,
            'change_date': '2023-06-15',
            'potential_cause': '境温度升高或冷却系统效率下降'
        },
        {
            'feature': 'ib_device_stat_bias_current_c_0',
            'initial_range': [8.5, 9.5],
            'current_range': [9.0, 10.0],
            'change_date': '2023-07-01',
            'potential_cause': '设备老化或负载增加'
        }
    ]
    memory_system.update_memory('long_term', 'data_distribution', 'changes', data_distribution_changes)

    # 测试记忆检索
    print("短期记忆类别:", memory_system.get_memory_categories('short_term'))
    print("长期记忆类别:", memory_system.get_memory_categories('long_term'))

    print("\n短期数据记忆:", memory_system.get_memory('short_term', 'data', 'current_feature_set'))
    print("\n长期数据记忆:", memory_system.get_memory('long_term', 'data', 'feature_list'))
    print("\n模型选择记忆:", memory_system.get_memory('short_term', 'model_selection', 'selected_models'))
    print("\n集成方法记忆:", memory_system.get_memory('short_term', 'ensemble_method', 'method'))

    # 测试记忆整合
    memory_system.consolidate_memory()

    # 测试相关记忆检索
    context = ['model', 'performance', 'selection', 'ensemble']
    relevant_memory = memory_system.get_relevant_memory(context)
    print("\n相关记忆:", json.dumps(relevant_memory, indent=2, ensure_ascii=False))

    # 添加新的长期记忆检索测试
    print("\n模型性能趋势:", memory_system.get_memory('long_term', 'model_performance', 'trends'))
    print("\n最佳特征组合:", memory_system.get_memory('long_term', 'feature_selection', 'best_combinations'))
    print("\n最优超参数范围:", memory_system.get_memory('long_term', 'hyperparameters', 'optimal_ranges'))
    print("\n故障模式与特征关联:", memory_system.get_memory('long_term', 'failure_patterns', 'feature_correlations'))
    print("\n最佳时间窗口大小:", memory_system.get_memory('long_term', 'data_preprocessing', 'optimal_time_window'))
    print("\n模型集成策略效果:", memory_system.get_memory('long_term', 'ensemble_strategies', 'effectiveness'))
    print("\n数据分布变化:", memory_system.get_memory('long_term', 'data_distribution', 'changes'))

if __name__ == "__main__":
    main()