import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from typing import List, Dict, Any, Tuple
from models.lstm_model import DynamicLSTM
from models.cnn_model import DynamicCNN
from models.gru_model import DynamicGRU
from models.transformer_model import DynamicTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from openai import OpenAI
from memory.memory import MemorySystem
from data_preprocessing.data_selection import load_data
import json
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class ModelAssemble:
    def __init__(self, memory_system: MemorySystem, candidate_models: List[str]):
        self.memory_system = memory_system
        self.candidate_models = candidate_models
        self.selected_models = []
        self.trained_models = []
        self.ensemble_model = None
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "infra-ppt-creativity"),
            base_url=os.getenv("LLM_API_BASE_URL", "https://api.lingyiwanwu.com/v1")
        )
        
        # 初始化长期记忆
        if not self.memory_system.get_memory('long_term', 'ensemble_strategies'):
            self.memory_system.update_memory('long_term', 'ensemble_strategies', [
                {'strategy': 'voting', 'average_performance_gain': 0},
                {'strategy': 'averaging', 'average_performance_gain': 0},
                {'strategy': 'stacking', 'average_performance_gain': 0}
            ])

    def select_models(self, data_features: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        print("开始选择模型...")
        recent_features = self.memory_system.get_memory('short_term', 'recent_features')
        model_performance = self.memory_system.get_memory('long_term', 'model_performance')
        recent_training = self.memory_system.get_memory('short_term', 'recent_training_selection')
        
        # 构建提示
        prompt = self._build_model_selection_prompt(data_features, recent_features, model_performance, recent_training)
        
        # 调用 LLM
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "你是一个数据科学家,擅长模型选择和超参数化。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000
            )
            llm_output = response.choices[0].message.content
            print("LLM 模型选择输出:")
            print(llm_output)
            selected_models_with_params, explanation = self._parse_llm_output_with_params(llm_output)
        except Exception as e:
            print(f"调用LLM时出错: {e}")
            selected_models_with_params = []
            explanation = "由于LLM调用失败,无法选择模型。"

        # 如果没有选择任何模型，选择默认模型
        if not selected_models_with_params:
            print("警告：没有选择任何模型。选择默认模型。")
            selected_models_with_params = [(self.candidate_models[0], {'input_size': data_features['input_size'], 'hidden_size': 64, 'output_size': 2})]
            explanation += " 使用默认模型。"

        self.selected_models = selected_models_with_params
        self.memory_system.update_memory('short_term', 'model_selection', {
            'selected_models': selected_models_with_params,
            'explanation': explanation
        })

        print(f"选择的模型和参数: {selected_models_with_params}")
        print(f"选择由: {explanation}")

        return self.selected_models

    def _build_model_selection_prompt(self, data_features, recent_features, model_performance, recent_training):
        # 将numpy的int64转换为Python的int
        data_features = {k: int(v) if isinstance(v, np.integer) else v for k, v in data_features.items()}
        
        prompt = f"""
        根据以下信息,推荐最适合的模型类型及其超参数:

        1. 数据特征:
        {json.dumps(data_features, indent=2)}

        2. 选择的特征:
        {json.dumps(recent_features, indent=2)}

        3. 模型性能趋势:
        {json.dumps(model_performance, indent=2)}

        4. 最近的训练选择:
        {json.dumps(recent_training, indent=2)}

        5. 候选模型:
        {self.candidate_models}

        请选择2-3个最适合的模型,并为每个模型提供合适的超参数。同时,请解释你的选择理由。

        请严格按照以下JSON格式输出你的选择:

        {{
            "selected_models": [
                {{
                    "name": "模型名称",
                    "params": {{
                        "param1": 值,
                        "param2": 值,
                        ...
                    }}
                }},
                ...
            ],
            "explanation": "选择理由"
        }}

        确保JSON格式正确,可以被直接解析。
        """
        return prompt

    def _parse_llm_output_with_params(self, llm_output: str) -> Tuple[List[Tuple[str, Dict[str, Any]]], str]:
        print("解析 LLM 输出:")
        print(llm_output)
        try:
            output_dict = json.loads(llm_output)
            selected_models_with_params = [(model['name'], model['params']) for model in output_dict['selected_models']]
            explanation = output_dict['explanation']
        except json.JSONDecodeError:
            print("警告：无法解析 LLM 输出为 JSON")
            selected_models_with_params = []
            explanation = "LLM 输出格式不正确，无法解析。"
        except KeyError as e:
            print(f"警告：LLM 输出缺少必要的键: {e}")
            selected_models_with_params = []
            explanation = f"LLM 输出缺少必要的信息: {e}"
        
        return selected_models_with_params, explanation

    def train_models(self, X_train, y_train):
        self.trained_models = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        for model_name, params in self.selected_models:
            print(f"\n训练模型: {model_name}")
            model = self._create_model(model_name, params)
            model.to(device)
            
            optimizer = optim.Adam(model.parameters(), lr=params.get('learning_rate', 0.001))
            criterion = nn.CrossEntropyLoss()
            
            for epoch in range(params.get('epochs', 10)):
                model.train()
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                
                self.trained_models.append((model_name, model))
            
        return self.trained_models

    def assemble_models(self):
        if not self.trained_models:
            raise ValueError("没有训练好的模型可供集成")
        
        def ensemble_predict(X):
            predictions = []
            for model_name, model in self.trained_models:
                if isinstance(model, torch.nn.Module):
                    model.eval()
                    with torch.no_grad():
                        pred = model(torch.FloatTensor(X)).numpy()
                else:
                    pred = model.predict_proba(X)
                predictions.append(pred)
            return np.mean(predictions, axis=0)
        
        return ensemble_predict

    def _voting_ensemble(self, X):
        # 实现投票集成方法
        pass

    def _stacking_ensemble(self, X):
        # 实现堆叠集成方法
        pass

    def _average_ensemble(self, X):
        print("使用平均集成方法")
        if not self.trained_models:
            print("警告：没有训练好的模型可供集成")
            return np.zeros(X.shape[0])  # 返回全零数组
        
        predictions = []
        for model_name, model in self.trained_models:
            print(f"获取 {model_name} 的预测")
            if isinstance(model, torch.nn.Module):
                with torch.no_grad():
                    pred = model(torch.FloatTensor(X)).numpy()
            else:
                pred = model.predict_proba(X)
            print(f"{model_name} 预测形状: {pred.shape}")
            predictions.append(pred)
        
        mean_pred = np.mean(predictions, axis=0)
        print(f"平均预测形状: {mean_pred.shape}")
        return mean_pred

    def _create_model(self, model_name: str, params: Dict[str, Any]):
        try:
            if model_name == 'LSTM':
                return DynamicLSTM(
                    input_size=params.get('input_size', 5),
                    hidden_size=params.get('hidden_size', 64),
                    num_layers=params.get('num_layers', 2),
                    output_size=params.get('output_size', 2)
                )
            elif model_name == 'GRU':
                return DynamicGRU(
                    input_size=params.get('input_size', 5),
                    hidden_size=params.get('hidden_size', 64),
                    num_layers=params.get('num_layers', 2),
                    output_size=params.get('output_size', 2)
                )
            elif model_name == 'CNN':
                return DynamicCNN(
                    input_size=params.get('input_size', 5),
                    num_channels=params.get('num_channels', [32, 64, 128]),
                    kernel_sizes=params.get('kernel_sizes', [3, 3, 3]),
                    output_size=params.get('output_size', 2)
                )
            elif model_name == 'Transformer':
                return DynamicTransformer(
                    input_size=params.get('input_size', 5),
                    d_model=params.get('d_model', 64),
                    nhead=params.get('nhead', 4),
                    num_layers=params.get('num_layers', 2),
                    output_size=params.get('output_size', 2)
                )
            else:
                raise ValueError(f"不支持的模型类型: {model_name}")
        except Exception as e:
            print(f"创建模型 {model_name} 时发生错误: {e}")
            print("使用默认LSTM模型")
            return DynamicLSTM(input_size=params.get('input_size', 5), hidden_size=64, num_layers=2, output_size=2)

    def _evaluate_model(self, model, X_test, y_test):
        if callable(model):
            y_pred = model(X_test)
        elif isinstance(model, torch.nn.Module):
            model.eval()
            with torch.no_grad():
                y_pred = model(torch.FloatTensor(X_test)).numpy()
        else:
            y_pred = model.predict(X_test)
        
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred_classes)
        precision = precision_score(y_test, y_pred_classes, average='weighted')
        recall = recall_score(y_test, y_pred_classes, average='weighted')
        f1 = f1_score(y_test, y_pred_classes, average='weighted')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    def _build_ensemble_prompt(self, model_names):
        prompt = f"""
        根据以下训练好的模型,推荐最佳的集成方法:
        {model_names}
        
        可选的集成方法包括:
        1. 投票法 (Voting)
        2. 平均法 (Averaging)
        3. 堆叠法 (Stacking)
        
        请选择一种方法并解释你的选择理由。
        """
        return prompt

    def _parse_ensemble_output(self, llm_output: str) -> Tuple[str, str]:
        print("解析 LLM 集成输出:")
        print(llm_output)
        lines = llm_output.split('\n')
        ensemble_method = "averaging"  # 默认方法
        explanation = ""
        for line in lines:
            if "选择的方法:" in line:
                ensemble_method = line.split(':')[1].strip().lower()
            elif "解释:" in line:
                explanation = line.split(':', 1)[1].strip()
        if not explanation:
            explanation = "LLM 没有提供明确的解，使用默认的平均集成方法。"
        return ensemble_method, explanation

    def _parse_llm_output_with_params(self, llm_output: str) -> Tuple[List[Tuple[str, Dict[str, Any]]], str]:
        print("解析 LLM 输出:")
        print(llm_output)
        lines = llm_output.split('\n')
        selected_models_with_params = []
        explanation = ""
        current_model = ""
        current_params = {}
        
        for line in lines:
            line = line.strip()
            if line.endswith(':'):
                if current_model and current_params:
                    selected_models_with_params.append((current_model, current_params))
                current_model = line[:-1]
                current_params = {}
            elif line.startswith('{') and line.endswith('}'):
                current_params = json.loads(line)
            elif line.startswith("解释:"):
                explanation = line.split(':', 1)[1].strip()
                if current_model and current_params:
                    selected_models_with_params.append((current_model, current_params))
        
        return selected_models_with_params, explanation

def main():
    print("开始主程序...")
    # 加载记忆系统
    memory_path = './src/memory/memory_test.json'
    loaded_memory = MemorySystem.load_from_json(memory_path)
    candidate_models = ['LSTM', 'CNN', 'GRU', 'Transformer']
    ensemble = ModelAssemble(loaded_memory, candidate_models)

    # 加载之前处理好的数据
    train_data_path = "./data/raw/processed/training_data/train_samples.jsonl"
    test_data_path = "./data/raw/processed/training_data/test_samples.jsonl"

    print("加载训练数据...")
    train_samples = load_data(train_data_path)
    X_train = np.array([sample.features for sample in train_samples])
    y_train = np.array([sample.label for sample in train_samples])

    print("加载测试数据...")
    test_samples = load_data(test_data_path)
    X_test = np.array([sample.features for sample in test_samples])
    y_test = np.array([sample.label for sample in test_samples])

    print(f"训练数据形状: {X_train.shape}")
    print(f"测试数据形状: {X_test.shape}")

    # 更新当前的时间窗口大小
    current_window_size = loaded_memory.get_memory('short_term', 'window_size', 'current')
    if current_window_size:
        # 安全地获取嵌列表中的值
        while isinstance(current_window_size, list) and current_window_size:
            current_window_size = current_window_size[0]
        if not isinstance(current_window_size, (int, float)):
            print("警告：法从记忆中获取有效的时间窗口大小")
            current_window_size = X_train.shape[1]
    else:
        current_window_size = X_train.shape[1]
    
    print(f"当前时间窗口大小: {current_window_size}")
    loaded_memory.update_memory('short_term', 'data', 'time_window', current_window_size)

    # 选择模型和超参数
    data_features = {
        "feature_count": int(X_train.shape[2]) if len(X_train.shape) > 2 else int(X_train.shape[1]),
        "sample_count": int(X_train.shape[0]),
        "time_steps": int(X_train.shape[1]) if len(X_train.shape) > 2 else 1,
        "input_size": int(X_train.shape[2]) if len(X_train.shape) > 2 else int(X_train.shape[1]),
        "label_distribution": dict(zip(map(int, np.unique(y_train)), map(int, np.unique(y_train, return_counts=True)[1])))
    }
    print("数据特征:")
    print(json.dumps(data_features, indent=2, ensure_ascii=False))
    
    selected_models = ensemble.select_models(data_features)

    # 训练模型
    trained_models = ensemble.train_models(X_train, y_train)
    
    if not trained_models:
        print("警告：没有成功训练的模型")
        return

    # 选择集成方法
    ensemble_model = ensemble.assemble_models()

    # 进行预测
    print("开始进行预测...")
    y_pred = ensemble_model(X_test)

    print(f"集成预测的形状: {y_pred.shape}")
    print(f"前几个集成预测值: {y_pred[:5]}")

    # 评估模型性能
    performance = ensemble._evaluate_model(ensemble_model, X_test, y_test)

    print(f"模型性能: {performance}")

    # 更新长期记忆中的模型性能趋势
    ensemble.memory_system.update_memory('long_term', 'model_performance', 'trends', {
        'Ensemble': {
            'accuracy_trend': performance['accuracy'],
            'f1_trend': performance['f1_score']
        }
    })

    # 保存更新后的记忆系统
    ensemble.memory_system.save_to_json(memory_path)
    print("记忆系统已更新并保存")

if __name__ == "__main__":
    main()

