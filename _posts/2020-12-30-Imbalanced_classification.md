---
title: Imbalanced data classification
author: Kwon Suncheol
categories: [ML]
tags: [Imbalanced_data,Churn_prediction,Classification,Data_sampling,Metric]
pin: false
math: true
comments: true
---

![Imbalanced_image](/assets/img/post_img/Imbalanced_image.jpg)_Cute imbalanced image[^1]_

<br>

 지도학습에서 분류 문제를 다룰 때 Imbalanced classification인 경우가 많았습니다. 대표적으로 이커머스 데이터 같은 경우 개별적인 고객의 이탈을 예측하는 모델을 만들 때 위의 문제를 발견할 수 있었습니다. 이탈 예측 태스크를 진행하면서 생각한 것들을 정리하는 시간을 가졌습니다.


## Imbalanced data
___________________

### Definition of Imbalanced data 

처음에는 데이터의 반응변수의 분포가 고르게 퍼진 Unbalanced data와 다르게 **Imbalanced data는 도메인 특성상 내재적인 이유로 처음부터 반응변수의 분포가 왜곡된 데이터로 정의**할 수 있습니다.  이러한 문제는 앞에서 언급한 '고객 이탈 예측', '이상치 탐지', '스팸 탐지'까지 다양한 상황에서 직면할 수 있습니다. 

|Case|심각한 불균형 O|심각한 불균형 X|충분한 데이터 샘플 O|충분한 데이터 샘플 X|
|:------:|:---:|:---:|:---:|:---:|
|case 1|<span style="color:red">O</span>|<span style="color:blue">X</span>|<span style="color:red">O</span>|<span style="color:blue">X</span>|
|case 2|<span style="color:red">O</span>|<span style="color:blue">X</span>|<span style="color:blue">X</span>|<span style="color:red">O</span>|
|case 3|<span style="color:blue">X</span>|<span style="color:red">O</span>|<span style="color:red">O</span>|<span style="color:blue">X</span>|
|case 4|<span style="color:blue">X</span>|<span style="color:red">O</span>|<span style="color:blue">X</span>|<span style="color:red">X</span>|

각각의 케이스마다 접근 방법을 다르게 할 필요성이 있지만 특히 case 2번[심각한 불균형이면서 충분한 데이터 샘플을 확보하지 못한 경우]같은 경우는 분류 모델을 구성하기 상당히 힘듭니다. 위의 테이블에서 우리는 데이터 자체의 크기, 노이즈(Noise)를 유발하는 데이터, 그리고 왜도가 높은 데이터 분포가 특정한 분류 문제를 어렵게 만듬을 알 수 있습니다. 

<br>

## Evaluation Metrics of Imbalanced data classification
___________________


### Three families of evaluation metrics of classification

다음의 논문[^2]에서 분류 문제에 대한 평가 지표를 다음과 같이 세 가지로 구분하였습니다.  
- Threshold Metrics
- Ranking Metrics
- Probability Metrics

<br/>

### Threshold Metrics

'Threshold Metrics'는 분류 예측 모델의 오분류를 정량화한 지표입니다. 다음과 같은 표를 많이 보셨을 겁니다. 

||Positive Prediction|Negative Prediction|
|:------:|:---:|:---:|
|**Positive Class**|TP(True Positive)|FN(False Negative)|
|**Negative Class**|FP(False Positive)|TN(True Negative)|

이커머스에서 이탈 예측 문제를 해당 표와 연결하면 다음과 같습니다.

||이탈로 예측|이탈이 아니라고 예측|
|:------:|:---:|:---:|
|**실제 이탈 O**|TP(True Positive)|FN(False Negative)|
|**실제 이탈 X**|FP(False Positive)|TN(True Negative)|

 사기 탐지(Fraud detection) 영역에서는 다음과 같은 그림으로 위의 표의 케이스를 응용하여 해석할 수 있습니다. 

![false-positive-test-diagram](/assets/img/post_img/false-positive-test-diagram.png)_false-positive test diagram in fraud detection[^3]_

<br>

$$
Accuracy = \frac {TP + TN}{TP+FN+FP+TN} 
$$

<br>

$$
Error = 1 - Accuracy = \frac {FP + FN}{TP+FN+FP+TN} 
$$

<br/>

### Sensitivity-Specificity Metrics   

 일반적인 분류 문제에서는 정확도를 많이 참고합니다. 하지만 사기 탐지 데이터와 같이 극단적으로 불균형된 데이터인 경우 정확도는 크게 의미가 없습니다. FN(False Negative)와 FP(False Postivie) 모두 예측 모델이 틀린 경우들이지만 사기 탐지인 경우 FN일 때가 FP인 경우보다 훨씬 치명적입니다.  <br/>
이것을 한번 위의 표와 엮으면 다음과 같은 지표를 고안하여 해석할 수 있습니다. 

<br/>

$$
Sensitivity = \frac {TP}{TP+FN} 
$$

<br>

$$
Specificity = \frac {TN}{FP+TN} 
$$

<br/> 

사기 탐지 같은 경우 민감도를 특이도보다 더 중요한 지표로 삼을 것입니다. 이 해석을 제가 마주한 문제에 비유하면 이탈을 하지 않는데 이탈을 하는 것으로 예측하는 것이 미래 시점에 이탈이 발생했는데 잘못 예측하는 것보다 덜 치명적일 것입니다.<br/>

만약 특이도와 민감도 두 가지 모두를 고려하고 싶을 때는 두 지표의 기하평균인 G-mean을 사용합니다. 

<br/> 

$$
G-mean = \sqrt{Sensitiviy \times Specificity}
$$

<br/> 

### Precision-Recall Metrics

수식상으로 Precision과 Recall을 살펴보면 다음과 같습니다.

<br/> 

$$
Precision = \frac {TP}{TP+FP} 
$$

<br/> 

$$
Recall = \frac {TP}{TP+FN} 
$$

<br/> 

 수식의 의미를 사기 탐지 태스크와 이탈 예측 태스크에 각각 맞추어서 살펴보겠습니다.  사기 탐지 태스크에서 정밀도의 의미는 해당 클래스가 '사기'로  예측하였을 때 실제로 사기일 비율을 의미합니다. 이와 유사하게 **이탈 예측 태스크에서 정밀도의 의미는 해당 클래스가 '이탈'로 예측하였을 때 실제로 '이탈'을 할 비율**을 뜻합니다.  
 <br/>
  재현율은 수식상으로 민감도와 같습니다. 사기 탐지 태스크에서 재현율은 해당 클래스가 실제로 '사기'일 때 사기로 예측할 확률을 말합니다. 이탈 예측 태스크에서는 실제로 '이탈'일 때 이탈로 예측할 비율을 의미합니다.    
 <br/> 
수식에서 알 수 있듯이 정밀도와 재현율은 일반적으로 다음과 같이 표현될 것입니다.

![Precision_Recall_Curve_image1](/assets/img/post_img/Precision_Recall_Curve_image1.png)_General Precision-Recall Curve[^4]_
 
 모델의 성능이 아주 뛰어나거나 안좋으면 다음과 같이 그려질 것입니다.
 
![Precision_Recall_Curve_image2](/assets/img/post_img/Precision_Recall_Curve_image2.png)_Best Precision-Recall Curve_ 

![Precision_Recall_Curve_image3](/assets/img/post_img/Precision_Recall_Curve_image3.png)_Worst Precision-Recall Curve_ 


<br>

## Data sampling
___________________

- 데이터 샘플링은 반응 변수의 분포를 유지시키기 위한 방법입니다. 예측 모델을 신중히 선택하는 만큼 다양한 데이터 샘플링 기법들 중 가장 자신의 문제 상황에 맞는 방법을 결정해야 합니다.
- 만약 데이터가 왜곡되어 있다면 많은 머신러닝 모델들은 다수를 차지하는 클래스에서 관측되는 예측 변수들의 가능도에 초점이 맞추어 질 것입니다.
- 이런 경우, 이탈 예측과 같이 소수 클래스와 예측 변수들의 관계를 포착하는 것이 중요한 태스크는 예측 모델을 구성할 때 심각한 문제에 직면할 수 있습니다.

### Random sampling

#### Random Undersampling
- 학습 데이터 세트에서 다수 클래스를 랜덤하게 선택한 후 지우는 샘플링 기법입니다.
- 불균형한 데이터이지만 소수를 차지하는 클래스가 충분한 관측치를 가지고 있는 경우 사용합니다.
- 다만 Undersampling 기법의 특성상 다수의 클래스에서 지워지는 데이터가 분류 경계를 짓는데 아주 중요할 경우 예측 모델 정확도가 떨어질 수 있습니다.  

<br/>

![random_undersampling_image](/assets/img/post_img/random_undersampling_image.png)_RUS_[^5]

#### Random Oversampling
- 소수 클래스의 데이터 중에서 일정한 수를 랜덤하게 선택하여 다수 클래스의 수만큼 복제하는 기법입니다.
- 왜곡된 분포에 크게 영향을 받는 머신러닝 모델에 적합하지만 소수의 클래스에 과적합이 발생하여 일반화에 악영향을 끼칠 수 있습니다.
- 만약 데이터가 엄청 클 경우 계산 속도가 오래 걸립니다.
- 이런 악영향을 방지하기 위해서 Raw data에 예측 모델과 Random Oversapling을 적용한 데이터에 적합시킨 예측 모델간의 성능을 비교할 필요가 있습니다.

<br/>

![random_oversampling_image](/assets/img/post_img/random_oversampling_image.pbm)_ROS_{: width="200" height="500"}[^6]

### Undersampling

- 다음의 상황을 가정해 봅니다.
	- 아래의 표는 각자의 성과 특성이 담긴 테이블입니다.
	- 다음과 같은 기준으로 사람들을 묶을 수 있을 것입니다.
		- **비슷한 특질이 있는 사람들 끼리 묶기!** 
		- **비슷하지 않는 특색을 가진 사람들을 구분하기!**
		- **첫번째 방법과 두번째 방법을 섞어서 묶기!**

|Feature|<span style="color:red">Lee</span>|<span style="color:red">Koo</span>|<span style="color:blue">Kwon</span>|<span style="color:blue">Kim</span>|<span style="color:blue">Ko</span>|<span style="color:black">Cho</span>|Oh
|:------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|**Feature_a**|O|O|O|O|O|X|X|
|**Feature_b**|O|O|X|X|X|O|O|
|**Feature_c**|X|X|O|O|O|X|X|

<br/>

Undersampling도 앞의 세 가지 경우와 비슷합니다. 구체적으로 다음과 같이 기법을 세분화할 수 있습니다.  
- 유지할 데이터를 선택하는 방법
	- Near Miss Undersampling
	- Condensed Nearest Neighbor Rule 	
- 제거할 데이터를 선택하는 방법
	- Tomek Links
	- ENN(Edited Nearest Neighbor Rule) 	
- 첫번째와 두번째를 적절히 섞은 방법
	- One-Sided Selection
		-  Tomek Links -> CNN
	- Neighborhood Cleansing Rule
		-  CNN -> ENN

### Mixed Sampling

- SMOTE + Tomek Links
- SMOTE + Edited Nearest Neighbor Rule

## Reference

[^1]: https://medium.com/@kr.vishwesh54/a-creative-way-to-deal-with-class-imbalance-without-generating-synthetic-samples-4cfad099d405
[^2]: https://www.math.ucdavis.edu/~saito/data/roc/ferri-class-perf-metrics.pdf
[^3]: https://www.appsflyer.com/blog/click-flooding-detection-false-positive-challenge/
[^4]: https://towardsdatascience.com/gaining-an-intuitive-understanding-of-precision-and-recall-3b9df37804a7
[^5]: https://www.researchgate.net/figure/llustration-of-random-undersampling-technique_fig3_343326638
[^6]: https://www.researchgate.net/figure/Random-Oversampling-of-Data_fig3_336305785
