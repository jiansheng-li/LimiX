import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
from sklearn.datasets import make_classification, make_regression
import warnings

from utils.inference_utils import auc_metric

warnings.filterwarnings('ignore')


class CatBoostTrainer:
    def __init__(self, task_type='classification', **kwargs):
        """
        初始化CatBoost训练器
        
        Parameters:
        task_type (str): 任务类型，'classification' 或 'regression'
        **kwargs: CatBoost模型的其他参数
        """
        self.task_type = task_type
        if task_type == 'classification':
            self.model = CatBoostClassifier(**kwargs)
        elif task_type == 'regression':
            self.model = CatBoostRegressor(**kwargs)
        else:
            raise ValueError("task_type must be 'classification' or 'regression'")
    
    def train(self, X_train, y_train, X_val=None, y_val=None, **fit_params):
        """
        训练CatBoost模型
        
        Parameters:
        X_train: 训练特征
        y_train: 训练标签
        X_val: 验证特征（可选）
        y_val: 验证标签（可选）
        **fit_params: fit方法的其他参数
        """
        # 准备训练数据
        if X_val is not None and y_val is not None:
            fit_params['eval_set'] = (X_val, y_val)
            fit_params['early_stopping_rounds'] = fit_params.get('early_stopping_rounds', 10)
        
        # 训练模型
        self.model.fit(X_train, y_train, **fit_params)
        
    def predict(self, X):
        """
        使用训练好的模型进行预测
        
        Parameters:
        X: 需要预测的特征
        
        Returns:
        预测结果
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        对于分类任务，预测样本属于各类别的概率
        
        Parameters:
        X: 需要预测的特征
        
        Returns:
        预测概率
        """
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
            # 对于二分类任务，计算AUC
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


def catboost_classification(X_train, y_train, X_test, y_test):
    """分类任务示例"""
    print("=== CatBoost 分类任务示例 ===")

    
    # 初始化分类器
    clf = CatBoostTrainer(
        task_type='classification',
        iterations=100,
        learning_rate=0.1,
        depth=6,
        verbose=0,
        random_seed=42
    )
    
    # 训练模型
    clf.train(X_train, y_train, X_test, y_test, verbose=0)
    
    # 评估模型
    metrics = clf.evaluate(X_test, y_test)
    return metrics



def catboost_regression(X_train, y_train, X_test, y_test):


    reg = CatBoostTrainer(
        task_type='regression',
        iterations=100,
        learning_rate=0.1,
        depth=6,
        verbose=0,
        random_seed=42
    )
    
    # 训练模型
    reg.train(X_train, y_train, X_test, y_test, verbose=0)
    
    # 评估模型
    metrics = reg.evaluate(X_test, y_test)
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R2: {metrics['r2']:.4f}")



if __name__ == '__main__':
    # 运行分类示例
    pass