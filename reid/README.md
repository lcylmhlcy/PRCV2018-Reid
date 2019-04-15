# Pedestrian ID Retrieval (PR-ID)

## Method
Refer to the paper **[PCB](https://arxiv.org/abs/1711.09349)** and **[github code](https://github.com/huanghoujing/beyond-part-models)**  
We add a brance at the following of feature maps extracted by resnet50. The branch is the same as the other sub-branches. This branch we added is to extrace global features. Combined with other local features, the model has a better performance.  

## Evaluation
[Interfaces doc and util codes for the competition of Large-scale Pedestrian Retrieval](https://github.com/dli2016/LSPR) 
- PR-ID-RAP: Pedestrian ID retrieval performance is measured according to the standard Pedestrian Recognition Performance Evaluation Index (mAP).
- System Evaluation (PR-ID-SYS): According to the comparison between the true value of the question and answer and the returned binary results, the F-Score of the ReID question and answer is calculated. At the same time, the detection rate of pedestrian detection is combined to calculate the final system performance indicators.

Most details: http://prcv.qyhw.net.cn/pages/20  

## Result
![img](https://github.com/lcylmhlcy/PRCV2018-Reid/raw/master/img/csv.png)  

1. The output csv is in folder **csv** and model dict is in folder **saved_model**, as follows:  
![img](https://github.com/lcylmhlcy/PRCV2018-Reid/raw/master/img/2.png) 

2. **csv** folder has three files:  
![img](https://github.com/lcylmhlcy/PRCV2018-Reid/raw/master/img/1.png) 
