import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
from memory.memory import MemorySystem
from data_preprocessing.data_selection import load_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from models.model_assemble import ModelAssemble

def load_selected_data(data_path):
    samples = load_data(data_path)
    train_samples = [sample for sample in samples if sample.split == 'train']
    test_samples = [sample for sample in samples if sample.split == 'test']
    
    if not train_samples or not test_samples:
        print("警告：没有找到带有 'train' 或 'test' 标记的样本。将随机划分数据。")
        train_samples, test_samples = train_test_split(samples, test_size=0.2, random_state=42)
        for sample in train_samples:
            sample.split = 'train'
        for sample in test_samples:
            sample.split = 'test'
    
    return train_samples, test_samples

def prepare_data_for_training(samples):
    X = np.array([sample.features for sample in samples])
    y = np.array([sample.label for sample in samples])
    return torch.FloatTensor(X), torch.LongTensor(y)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # 验证
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {loss.item():.4f}, '
              f'Val Loss: {val_loss/len(val_loader):.4f}, '
              f'Val Acc: {correct/total:.4f}')

def evaluate_model(model, test_loader, device):
    if callable(model):
        all_preds = []
        all_labels = []
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X.numpy())
            _, predicted = torch.tensor(outputs).max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    else:
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def validate_selected_models(selected_models):
    if not isinstance(selected_models, list):
        return False
    for item in selected_models:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return False
        if not isinstance(item[0], str) or not isinstance(item[1], dict):
            return False
    return True

def main():
    # 加载记忆系统
    memory_path = './src/memory/memory_test.json'
    memory_system = MemorySystem.load_from_json(memory_path)

    # 加载数据
    data_path = "./data/raw/processed/selected_samples.jsonl"
    train_samples, test_samples = load_selected_data(data_path)

    # 准备数据
    X_train, y_train = prepare_data_for_training(train_samples)
    X_test, y_test = prepare_data_for_training(test_samples)

    # 创建数加载器
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 获取模型选择和超参数
    selected_models = memory_system.get_memory('short_term', 'selected_models')
    if not selected_models or not validate_selected_models(selected_models):
        print("告：未找到有效的模���选择信息，使用默认模型")
        selected_models = [('LSTM', {'input_size': X_train.shape[2], 'hidden_size': 64, 'num_layers': 2})]
    
    try:
        # 创建模型集成
        model_assemble = ModelAssemble(memory_system, [model[0] for model in selected_models])
        
        # 训练每个模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for model_name, params in selected_models:
            model = model_assemble._create_model(model_name, params)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            print(f"训练模型: {model_name}")
            train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10, device=device)
            
            # 评估模型
            performance = evaluate_model(model, test_loader, device)
            print(f"{model_name} 模型性能: {performance}")
            
            # 更新记忆系统
            memory_system.update_memory('short_term', f'{model_name}_performance', performance)

        # 训练集成模型
        ensemble_model = model_assemble.assemble_models()
        ensemble_performance = evaluate_model(ensemble_model, test_loader, device)
        print(f"集成模型性能: {ensemble_performance}")
        memory_system.update_memory('short_term', 'ensemble_performance', ensemble_performance)
    
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        print("使用默认模型进行训练")
        # 这里可以添加使用默认模型的训练逻辑

    # 保存更新后的记忆系统
    memory_system.save_to_json(memory_path)
    print(f"更新后的记忆系统已保存至 {memory_path}")

if __name__ == "__main__":
    main()
