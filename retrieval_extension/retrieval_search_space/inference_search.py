import optuna
from typing import Literal
from retrieval_extension.retrieval_search_space.init_search_space import generate_search_space
import numpy as np
import torch

from utils.inference_utils import auc_metric


class RetrievalSearchHyperparameters:
    def __init__(self, args,trainX,trainy,testX,testy,attention_score):
        self.args = args
        self.study=optuna.create_study(direction="maximize")
        self.trainX=trainX
        self.trainy=trainy
        self.testX=testX
        self.testy=testy
        self.attention_score=attention_score


    def search(self,method,metric:Literal["AUC","accuracy","f1"]="AUC",n_trials:int=1000):
        self.study.optimize(lambda trial: self.optuna_inference(trial,method,metric), n_trials=n_trials)
        best_params = self.study.best_params
        print(f"best_params: {best_params}")
        print(f"best metric on vaildation: {self.study.best_value}")
        return best_params,self.study.best_value



    def optuna_inference(self,trial,method,metric:Literal["AUC","accuracy","f1"]="accuracy"):
        param=generate_search_space(trial,self.args)
        output = method.inference(self.trainX, self.trainy, self.testX, attention_score=self.attention_score,**param)
        output = output[:, :len(np.unique(self.trainy))].float()
        outputs = torch.nn.functional.softmax(output, dim=1)

        output = outputs.float().cpu().numpy()
        prediction_ = output / output.sum(axis=1, keepdims=True)
        roc = auc_metric(self.testy, prediction_)

        return float(roc)



