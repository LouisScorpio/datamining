#coding=utf-8
'''
Created on 2017年11月24日

@author: Lu.Yi
'''
'''
在scikit-learn库中，有AdaBoostRegression（回归）和AdaBoostClassifier（分类）两个
在对和AdaBoostClassifier进行调参时，主要是对两部分进行调参：1）AdaBoost框架调参；2）弱分类器调参
'''

#导包
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier  

#载入数据，sklearn中自带的iris数据集
iris=load_iris()

'''
AdaBoostClassifier参数解释
base_estimator:弱分类器，默认是CART分类树：DecisionTressClassifier
algorithm：在scikit-learn实现了两种AdaBoost分类算法，即SAMME和SAMME.R，
           SAMME就是原理篇介绍到的AdaBoost算法，指Discrete AdaBoost
           SAMME.R指Real AdaBoost，返回值不再是离散的类型，而是一个表示概率的实数值，算法流程见后文
                            两者的主要区别是弱分类器权重的度量，SAMME使用了分类效果作为弱分类器权重，SAMME.R使用了预测概率作为弱分类器权重。
           SAMME.R的迭代一般比SAMME快，默认算法是SAMME.R。因此，base_estimator必须使用支持概率预测的分类器。
loss：这个只在回归中用到，不解释了
n_estimator:最大迭代次数，默认50。在实际调参过程中，常常将n_estimator和学习率learning_rate一起考虑
learning_rate:每个弱分类器的权重缩减系数v。f_k(x)=f_{k-1}*a_k*G_k(x)。较小的v意味着更多的迭代次数，默认是1，也就是v不发挥作用。
另外的弱分类器的调参，弱分类器不同则参数不同，这里不详细叙述
'''
#构建模型
clf=AdaBoostClassifier(n_estimators=100)  #弱分类器个数设为100
scores=cross_val_score(clf,iris.data,iris.target)
print(scores.mean())