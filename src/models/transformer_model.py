import torch
import torch.nn as nn

class DynamicTransformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, dim_feedforward, output_size):
        super(DynamicTransformer, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward),
            num_layers
        )
        self.fc = nn.Linear(d_model, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # 调整维度顺序以适应Transformer
        x = self.transformer(x)
        x = x.mean(dim=0)  # 对序列取平均
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

