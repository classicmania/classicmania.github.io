---
title: imbalanced data classification part2
author: Kwon Suncheol
categories: [ML]
tags: [Imbalanced_data,Churn_prediction,Classification,Cost_sensitive,Evaluation_metric]
pin: false
math: true
comments: true
---

<br/>

![weight_image](/assets/img/post_img/imbalanced_data_classification2/weight_image.jpg)

<br/>

## Cost-Sensitive Learning

[지난 글](#imbalanced_data_classification_part1)에 이어서 이탈 유무 및 LTV를 예측하는 태스크를 진행할 때 '불균형 데이터를 어떻게 다룰 것인가'에 관하여 숙고하였던 내용을 연속하여 다루겠습니다.  

데이터 자체의 크기, 노이즈를 유발하는 데이터, 왜도가 높은 데이터 분포 문제를 해결 하기 위해서 분류와 실수로 표현되는 확률간의 어려움을 연결하는 Threshold와 관련된 다양한 분류 모델의 Evaluation Metric을 자신의 문제 상황에 따라 적용할 수 있습니다. 또한 

