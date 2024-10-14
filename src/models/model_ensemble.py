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
from sklearn.metrics import accuracy_score
import numpy as np
from openai import OpenAI
from memory.memory import MemorySystem
from data_preprocessing.data_selection import load_data
import json

class ModelEnsemble:
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

    def select_models(self, data_features: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        context = ['model_performance', 'predictions', 'recent_training_selection', 'feedback']
        relevant_memory = self.memory_system.get_relevant_memory(context)
        
        model_performance = relevant_memory.get('long_term', {}).get('model_performance', [])
        recent_predictions = relevant_memory.get('short_term', {}).get('predictions', [])
        recent_training_selection = relevant_memory.get('short_term', {}).get('recent_training_selection', [])
        feedback = relevant_memory.get('short_term', {}).get('feedback', [])

        prompt = f"""
        根据以下数据特征和历史信息,推荐最适合的模型类型及其超参数:
        
        数据特征:
        {data_features}
        
        候选模型包括:
        {self.candidate_models}
        
        历史模型性能:
        {json.dumps(model_performance, ensure_ascii=False, indent=2)}
        
        最近的预测:
        {json.dumps(recent_predictions, ensure_ascii=False, indent=2)}
        
        最近的训练数据选择:
        {json.dumps(recent_training_selection, ensure_ascii=False, indent=2)}
        
        相关反馈:
        {json.dumps(feedback, ensure_ascii=False, indent=2)}
        
        请选择最适合的模型及其超参数,并解释你的选择理由。对于每个选择的模型，请提供以下格式的超参数建议：
        
        模型名: {
            "param1": value1,
            "param2": value2,
            ...
        }
        
        解释: 你的解释文本
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "你是一个数据科学家，擅长模型选择和超参数优化。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000
            )
            llm_output = response.choices[0].message.content
            selected_models_with_params, explanation = self._parse_llm_output_with_params(llm_output)
        except Exception as e:
            print(f"调用LLM时出错: {e}")
            selected_models_with_params = [(model, {}) for model in self.candidate_models[:2]]
            explanation = "由于LLM调用失败，默认选择了前两个模型，使用默认参数。"

        self.selected_models = selected_models_with_params
        self.memory_system.update_memory('short_term', 'model_selection', {
            'selected_models': selected_models_with_params,
            'explanation': explanation
        })

        return self.selected_models

    def _parse_llm_output_with_params(self, llm_output: str) -> Tuple[List[Tuple[str, Dict[str, Any]]], str]:
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

    def train_models(self, X_train, y_train):
        self.trained_models = []
        for model_name, params in self.selected_models:
            model = self._create_model(model_name, params)
            model.fit(X_train, y_train)
            self.trained_models.append((model_name, model))
            
            # 更新模型性能到记忆系统
            performance = self._evaluate_model(model, X_train, y_train)
            self.memory_system.update_memory('long_term', 'model_performance', {
                'model': model_name,
                'params': params,
                'performance': performance
            })
        
        return self.trained_models

    def _evaluate_model(self, model, X, y):
        # 这里可以添加更复杂的评估逻辑
        if isinstance(model, torch.nn.Module):
            with torch.no_grad():
                y_pred = model(torch.FloatTensor(X)).numpy()
        else:
            y_pred = model.predict(X)
        return accuracy_score(y, (y_pred > 0.5).astype(int))

    def ensemble_models(self):
        model_names = [name for name, _ in self.trained_models]
        prompt = f"""
        根据以下训练好的模型,推荐最佳的集成方法:
        {model_names}
        
        可选的集成方法包括:
        1. 投票法 (Voting)
        2. 平均法 (Averaging)
        3. 堆叠法 (Stacking)
        
        请选择一种方法并解释你的选择理由。
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "你是一个专门从事模型集成的数据科学家。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300
            )
            llm_output = response.choices[0].message.content
            ensemble_method, explanation = self._parse_ensemble_output(llm_output)
        except Exception as e:
            print(f"调用LLM时出错: {e}")
            ensemble_method = "averaging"
            explanation = "由于LLM调用失败，默认使用平均集成方法。"

        if ensemble_method == "voting":
            self.ensemble_model = self._voting_ensemble
        elif ensemble_method == "stacking":
            self.ensemble_model = self._stacking_ensemble
        else:  # 默认使用平均集成
            self.ensemble_model = self._average_ensemble

        self.memory_system.update_memory('short_term', 'ensemble_method', {
            'method': ensemble_method,
            'explanation': explanation
        })

        return self.ensemble_model

    def _parse_ensemble_output(self, llm_output: str) -> Tuple[str, str]:
        lines = llm_output.split('\n')
        ensemble_method = ""
        explanation = ""
        for line in lines:
            if line.startswith("选择的方法:"):
                ensemble_method = line.split(':')[1].strip().lower()
            elif line.startswith("解释:"):
                explanation = line.split(':', 1)[1].strip()
        return ensemble_method, explanation

    def _voting_ensemble(self, X):
        # 实现投票集成方法
        pass

    def _stacking_ensemble(self, X):
        # 实现堆叠集成方法
        pass

    def _average_ensemble(self, X):
        predictions = []
        for _, model in self.trained_models:
            if isinstance(model, torch.nn.Module):
                with torch.no_grad():
                    pred = model(torch.FloatTensor(X)).numpy()
            else:
                pred = model.predict_proba(X)
            predictions.append(pred)
        return np.mean(predictions, axis=0)

    def _create_model(self, model_name: str, params: Dict[str, Any]):
        if model_name == 'LSTM':
            return DynamicLSTM(**params)
        elif model_name == 'CNN':
            return DynamicCNN(**params)
        elif model_name == 'GRU':
            return DynamicGRU(**params)
        elif model_name == 'Transformer':
            return DynamicTransformer(**params)
        else:
            raise ValueError(f"Unknown model type: {model_name}")

def main():
    # 加载记忆系统
    memory_path = './src/memory/memory_test.json'
    loaded_memory = MemorySystem.load_from_json(memory_path)
    candidate_models = ['LSTM', 'CNN', 'GRU', 'Transformer']
    ensemble = ModelEnsemble(loaded_memory, candidate_models)

    # 加载之前处理好的数据
    train_data_path = "./data/raw/processed/training_data/train_samples.jsonl"
    test_data_path = "./data/raw/processed/training_data/test_samples.jsonl"

    # 读取训练数据
    train_samples = load_data(train_data_path)
    X_train = np.array([sample.features for sample in train_samples])
    y_train = np.array([sample.label for sample in train_samples])

    # 读取测试数据
    test_samples = load_data(test_data_path)
    X_test = np.array([sample.features for sample in test_samples])
    y_test = np.array([sample.label for sample in test_samples])

    # 选择模型
    data_features = {"shape": X_train.shape, "type": "time_series", "target": "binary"}
    selected_models = ensemble.select_models(data_features)
    print(f"选择的模型: {selected_models}")

    # 训练模型
    trained_models = ensemble.train_models(X_train, y_train)
    print(f"训练的模型: {[name for name, _ in trained_models]}")

    # 集成模型
    ensemble_model = ensemble.ensemble_models()
    print(f"集成方法: {type(ensemble_model).__name__}")

    # 评估集成模型
    y_pred = ensemble_model(X_test)
    accuracy = accuracy_score(y_test, (y_pred > 0.5).astype(int))
    print(f"集成模型准确率: {accuracy}")

    # 保存更新后的记忆
    loaded_memory.save_to_json(memory_path)
    print("更新后的记忆系统已保存")

if __name__ == "__main__":
    main()