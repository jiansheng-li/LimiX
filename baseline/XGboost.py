import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
from sklearn.datasets import make_classification, make_regression
import warnings

from utils.inference_utils import auc_metric

warnings.filterwarnings('ignore')


class XGBoostTrainer:
    def __init__(self, task_type='classification', **kwargs):
        """
        初始化XGBoost训练器
        
        Parameters:
        task_type (str): 任务类型，'classification' 或 'regression'
        **kwargs: XGBoost模型的其他参数
        """
        self.task_type = task_type
        if task_type == 'classification':
            self.model = xgb.XGBClassifier(**kwargs)
        elif task_type == 'regression':
            self.model = xgb.XGBRegressor(**kwargs)
        else:
            raise ValueError("task_type must be 'classification' or 'regression'")
    
    def train(self, X_train, y_train, **fit_params):


        self.model.fit(X_train, y_train, **fit_params)
        
    def predict(self, X):

        return self.model.predict(X)
    
    def predict_proba(self, X):

        if self.task_type == 'classification':
            return self.model.predict_proba(X)
        else:
            raise ValueError("predict_proba is only available for classification tasks")
    
    def evaluate(self, X_test, y_test):
        """
        评估模型性能
        
        Parameters:
        X_test: 测试特征
        y_test: 测试标签
        
        Returns:
        评估指标字典
        """
        y_pred = self.predict(X_test)
        
        if self.task_type == 'classification':
            accuracy = accuracy_score(y_test, y_pred)
            if len(np.unique(y_test)) == 2:
                y_proba = self.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_proba)
                return {'acc': accuracy, 'auc': auc}
            else:
                y_proba = self.predict_proba(X_test)
                roc = auc_metric(y_test, y_proba)
                return {'acc': accuracy, 'auc': roc}
        else:  # regression
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)
            return {'mse': mse, 'rmse': rmse, 'r2': r2}


def xgboost_classification(X_train, y_train, X_test, y_test):
    
    # 初始化分类器
    clf = XGBoostTrainer(
        task_type='classification',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        verbosity=0
    )

    # 训练模型
    clf.train(X_train, y_train, verbose=0)
    

    metrics = clf.evaluate(X_test.cpu(), y_test.cpu())
    return metrics




def demo_regression(X_train, y_train, X_test, y_test):

    reg = XGBoostTrainer(
        task_type='regression',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        verbosity=0
    )
    
    # 训练模型
    reg.train(X_train, y_train, verbose=0)
    
    # 评估模型
    metrics = reg.evaluate(X_test, y_test)
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R2: {metrics['r2']:.4f}")



if __name__ == '__main__':
    pass
