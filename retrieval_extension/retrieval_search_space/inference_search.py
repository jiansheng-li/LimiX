import optuna
from typing import Literal

from sklearn.metrics import f1_score, precision_score

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


    def search(self,method,metric:Literal["AUC","accuracy","f1","precision"]="AUC",n_trials:int=1000):
        self.study.optimize(lambda trial: self.optuna_inference(trial,method,metric), n_trials=n_trials)
        best_params = self.study.best_params
        print(f"best_params: {best_params}")
        print(f"best metric on vaildation: {self.study.best_value}")
        return best_params,self.study.best_value



    def optuna_inference(self,trial,method,metric:Literal["AUC","accuracy","f1","precision"]="accuracy"):
        param=generate_search_space(trial,self.args)
        print(f"current params: {param}")
        try:
            output = method.inference(self.trainX, self.trainy, self.testX, attention_score=self.attention_score,device_id=self.args["device_id"],**param)
        except:
            output = method.predict(self.trainX, self.trainy, self.testX, attention_score=self.attention_score,device_id=self.args["device_id"],**param)
        output = output[:, :len(np.unique(self.trainy))].float()
        outputs = torch.nn.functional.softmax(output, dim=1)

        output = outputs.float().cpu().numpy()
        prediction_ = output / output.sum(axis=1, keepdims=True)
        if metric=="AUC":
            return float(auc_metric(self.testy, prediction_))
        elif metric=="accuracy":
            return float(np.mean(np.argmax(output, axis=1) == self.testy))
        elif metric=="f1":
            return float(f1_score(self.testy, np.argmax(output, axis=1), average='macro'))
        elif metric=="precision":
            return float(precision_score(self.testy, np.argmax(output, axis=1), average="binary"))




