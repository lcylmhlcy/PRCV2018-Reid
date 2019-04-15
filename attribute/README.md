# Pedestrian Attribute Retrieval (PR-A) 

## Method
1. Refer to the paper **[Improving Person Re-identification by Attribute and Identity Learning](https://arxiv.org/abs/1703.07220)**。
2. Use Ensemble learning, including nine models. [code](https://github.com/lcylmhlcy/PRCV2018-Reid/raw/master/attribute/model/Stacking_CNN.py)

## Evaluation
[Interfaces doc and util codes for the competition of Large-scale Pedestrian Retrieval](https://github.com/dli2016/LSPR) 
- PR-A-RAP: Comparing the ranking results of the samples returned according to the attribute query conditions with the real values, calculating the standard mAP index value as the performance evaluation index of pedestrian attribute retrieval.
- System Assessment (PR-A-SYS): Comparing the true value of the question and answer with the result of the binary value returned, the F-Score of attribute question and answer is calculated, and the final system performance index is calculated with the detection rate of pedestrian detection.

Most details: http://prcv.qyhw.net.cn/pages/20  

## Result
![img](https://github.com/lcylmhlcy/PRCV2018-Reid/raw/master/img/csv.png)  

1. The output csv is in folder **csv** and model dict is in folder **saved_model**, as follows:  
![img](https://github.com/lcylmhlcy/PRCV2018-Reid/raw/master/img/3.png) 

2. **csv** folder has only csv file.
