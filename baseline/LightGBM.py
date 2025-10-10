import lightgbm as lgb
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score

from utils.inference_utils import auc_metric


class LightGBMTrainer:
    def __init__(self, task_type='classification', **kwargs):
        """
        初始化LightGBM训练器
        :param task_type: 任务类型，'classification' 或 'regression'
        :param kwargs: LightGBM模型的其他参数
        """
        self.task_type = task_type
        self.model = None
        self.params = kwargs
        if self.task_type == 'classification':
            self.params.setdefault('objective', 'binary')
            self.params.setdefault('metric', 'binary_logloss')
        elif self.task_type == 'regression':
            self.params.setdefault('objective', 'regression')
            self.params.setdefault('metric', 'l2')
        else:
            raise ValueError("task_type must be 'classification' or 'regression'")

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        训练LightGBM模型
        :param X_train: 训练特征
        :param y_train: 训练标签
        :param X_val: 验证特征
        :param y_val: 验证标签
        :param kwargs: 训练参数
        """
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_sets = [train_data]
        valid_names = ['train']

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('valid')

        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
            **kwargs
        )

    def predict(self, X):
        """
        使用训练好的模型进行预测
        :param X: 特征数据
        :return: 预测结果
        """
        if self.model is None:
            raise ValueError("Model is not trained yet. Please train the model first.")

        if self.task_type == 'classification':
            return self.model.predict(X)
        else:
            return self.model.predict(X)

    def evaluate(self, X, y):
        """
        评估模型性能
        :param X: 特征数据
        :param y: 真实标签
        :return: 评估指标
        """
        if self.model is None:
            raise ValueError("Model is not trained yet. Please train the model first.")

        y_pred = self.predict(X)

        if self.task_type == 'classification':
            accuracy = accuracy_score(y, y_pred)
            if len(np.unique(y)) == 2:
                y_proba = self.predict_proba(X)[:, 1]
                auc = roc_auc_score(y, y_proba)
                return {'acc': accuracy, 'auc': auc}
            else:
                y_proba = self.predict_proba(X)
                roc = auc_metric(y, y_proba)
                return {'acc': accuracy, 'auc': roc}
        else:
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            return {'mse': mse, 'r2': r2}


def lightgbm_classification(X_train, y_train, X_test, y_test):
    trainer = LightGBMTrainer(task_type='classification', num_leaves=128, learning_rate=0.05, n_estimators=100)
    trainer.train(X_train, y_train, X_test, y_test, num_boost_round=128)

    metrics = trainer.evaluate(X_test, y_test)
    return metrics


def lightgbm_regression(X_train, y_train, X_test, y_test):
    """
    回归任务示例
    """

    trainer = LightGBMTrainer(task_type='regression', num_leaves=31, learning_rate=0.05, n_estimators=100)
    trainer.train(X_train, y_train, X_test, y_test, num_boost_round=100, early_stopping_rounds=10, verbose_eval=10)

    # 评估模型
    metrics = trainer.evaluate(X_test, y_test)
    print(f"Regression Results:")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"R2 Score: {metrics['r2']:.4f}")


if __name__ == '__main__':
    pass