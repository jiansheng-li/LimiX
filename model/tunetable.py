import torch
import torch.nn as nn
import torch.optim as optim


class TuneTables(nn.Module):
    def __init__(self, p=5,e=192,f=10, num_classes=10):
        super().__init__()
        self.tune_X = nn.Parameter(torch.randn(1,p,f,e) * 0.01)
        self.labels = torch.tensor([i % num_classes for i in range(p)]).unsqueeze(0).to("cuda")
        self.tune_y = nn.Embedding(num_classes,e)
        self.p=p
    def forward(self, embedding_X, embedding_y,only_prompt:bool=False,eval_pos:int=0):
        if only_prompt:
            modifiedX = torch.cat([self.tune_X, embedding_X[:,eval_pos:]], dim=1)
            modifiedy = torch.cat([self.tune_y(self.labels), embedding_y[:,eval_pos:]], dim=1)
        else:
            modifiedX = torch.cat([self.tune_X, embedding_X], dim=1)
            modifiedy = torch.cat([self.tune_y(self.labels), embedding_y], dim=1)
        return modifiedX, modifiedy


# 示例用法
if __name__ == "__main__":
    pass
