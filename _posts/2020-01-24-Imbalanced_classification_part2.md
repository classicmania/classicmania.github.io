---
title: Imbalanced data classification part2
author: Kwon Suncheol
categories: [ML]
tags: [Imbalanced_data,Churn_prediction,Classification,Cost_sensitive,Evaluation_metric,Calibration,Probability_scoring,Brier_score,Logloss_score]
pin: false
math: true
comments: true
---

<br/>

![weight_image2](/assets/img/post_img/weight_image2.jpg)

<br/>

지난 포스트에 이어서 고객의 이탈 유무와 LTV를 예측하는 태스크를 진행할 때 **'불균형 데이터를 어떻게 다룰 것인가'에 관하여 숙고하였던 내용을 다루겠습니다.** 

불균형 데이터는 데이터 자체의 크기, 노이즈를 유발하는 데이터, 왜도가 높은 데이터 분포 문제와 관련이 깊습니다. 지난 포스트에서 우리는 명쾌한 클래스의 분류(Ex : 이탈 또는 비이탈)와 실수로 표현되는 확률(Ex : 이탈 확률)간의 관계를 연결하는 Threshold와 관련된 다양한 Evaluation Metric이 존재하는 것을 확인할 수 있었습니다. 
또한 기업 내부의 KPI와 집중하는 고객군이 어느 군(EX : 미래에 이탈을 할 고객군 VS 미래에 이탈을 하지 않을 고객군)인지에 따라 집중할 평가 지표가 다릅니다.


<br/>

## Probability Scoring
___________________

만약 비즈니스 관계자들과 유관부서가 명확한 클래스의 분류값이 아닌 확률값 자체를 원한다면 어떻게 대응해야 할까요?  
이런 경우 'Threhold Metrics'이 아닌 새로운 확률값에 최적화된 Metric이 필요합니다. 확률 지표는 실제로 알려진 클래스의 확률 분포와 모델이 예측한 클래스의 일치 정도를 표현합니다. 실제 예측된 확률값은 물론 특정한 작업이 필요합니다. 이 작업은 뒤에서 보다 자세히 다루겠습니다. 먼저 예측된 확률을 평가하는 대표적인 두 가지 지표들을 살펴보겠습니다.

### Logarithmic Loss Score
___________________

명확한 이해를 위해서 고전적인  '동전 던지기' 상황을 끌어오겠습니다. 이 상황은 다음과 같은 특징을 가집니다.
- 일반적으로 한번 시도합니다.
- 동전을 던질 때 나올 수 있는 경우를 X라 할 때, X는 셀 수 있는 유한개의 경우의 수를 가지면서 각 사건이 독립일 것입니다. 
- 다시 말해서 X는 앞(Head) 또는 뒤(Tail)일 것입니다. 그리고 각 개별 사건은 다른 사건들에 영향을 주지 않습니다.  
- \\(P(H) + P(T) = 1\\), where \\(P(H) = p\\)  

<br/>

위의 특징들은 다음과 같은 식으로 압축됩니다.

<br/>

$$
X_{i} \sim iid \,Bernolli(p)
$$

<br/>

또한 확률 질량 함수는 다음과 같습니다.

<br/>

$$
P(X=x) = \begin{cases}
  p  & if \, X = Head \\
  1-p & if \, X = Tail \\
  0 & \text{otherwhise}
\end{cases}
$$


좀 더 간결히 표현해보겠습니다.

$$
P(X=x) =
\begin{cases}
p^{x} \, \times \, (1-p)^{1-x}  & \text{if $X = Tail \, or \, Head$ } \\
0 & \text{otherwise}
\end{cases}
$$

<br/>

일정한 분포를 가지는 장기간의 R.V 산술 평균값을 확률 분포의 평균이라 정의할 때 특정 사건의 가능도는 아래와 같습니다.


$$ \begin{aligned} E[X]  &= 
\sum_{x \in \mathbb{R}} {x \, \times \, P(X=x)} \\ &=
1  \, \times \, P(X=1) + 0 \, \times \, P(X=0) \\ &=
1  \, \times \, p + 0 \, \times \, (1-p) \\ &= 
p \end{aligned}$$

<br/>

명시된 것들을 이탈 태스크에 적용해 보겠습니다. 예측 모델이 주어진 데이터를 통해  이탈 유무를 예측하는 것은 배반 사건이면서 독립 사건을 예측하는 경우입니다.

<br/>

$$
P(Y=y|X) = \begin{cases}
  \hat{p}  & if \, y = \text{Churn} \\
  1-\hat{p} & if \, y = \text{Not Churn} \\
\end{cases}
$$

<br/>

표본 공간에서 주어진 데이터의 값들의 함수는 주어진 샘플과 같은 값을 가지는 R.V의 상대적인 가능도를 제공하는 역할을 하고 그 함수를 확률 밀도 함수라 명명합니다.  

좋은 분류 모델은 \\( \hat{p}\\)와 \\( p\\)간의 차이가 크지 않습니다. 다시 말해서 \\( p\\)에 가장 가까운 추정치인 \\( \hat{p}\\)값이 높을 수록 성능이 우수한 모델입니다.

<br/>

$$ \begin{aligned} L(\theta)  &= 
P(Y|X;\theta) \\ &=
\prod_{i=1}^{N(x)}P(y^{i} | x^{i}) 
\end{aligned}$$

<br/>

결국 \\(\theta\\)에 대한 가능도는 동전 던지기와 같은 상황이므로 베르누이 분포를 따릅니다. 

<br/>

$$
P(y^{i} | x^{i})  \sim iid \,Bernolli(\theta)
$$

<br/>

확률 밀도 함수(f)는 각 사건이 독립이므로 개별 확률 밀도 함수의 곱으로 표현할 수 있습니다. 

$$
f(y_{1},y_{2},...,y_{N(x)}|\hat{p}) = p^{y_{1}}  (1-\hat{p})^{1-y_{1}} \,\times  \hat{p}^{y_{2}}  (1-\hat{p})^{1-y_{2}} \,\times  \, ... \,\times \hat{p}^{y_{N(x)}}  (1-\hat{p})^{1-y_{N(x)}}
$$

<br/>

가능도 함수를 이탈 확률(\\( p\\))로 편미분한 기울기값이 0일 때의 X값은 최적의 P일 것입니다. Log 함수가 단조 함수라는 것을 이용하여 가능도 함수를 변형하겠습니다.

<br/>

$$
\begin{aligned} Log \,L  &= 
Log(\prod_{i =1}^{N(x)}\hat{p}^{y_{i}}  (1-\hat{p})^{1-y_{i}}) \\ &=
\sum_{i=1}^{N(x)}log(\hat{p}^{y_{i}}  (1-\hat{p})^{1-y_{i}}) \\ &=
\sum_{i=1}^{N(x)} y_{i}\,log(\hat{p}) + (1-y_{i}) \, log(1-\hat{p})\\ 
 \end{aligned}
$$

<br/>

위의 식에서는 gradient ascent 방식이지만 좌우변에 -를 붙히면 log-likelihood 함수를 최소화하는 경사 하강법을 적용할 수 있습니다. 

<br/>

$$
- Log \,L = \sum_{i=1}^{N(x)} -y_{i}\,log(\hat{p}) + (-1+y_{i}) \, log(1-\hat{p})\\ 
$$

<br/>

평균적인 loss function 값을 갖는 Cost function은 다음과 같을 것입니다.

<br/>

$$
Average(Log Loss) = \frac{1}{N(x)}\times  \sum_{i=1}^{N(x)} -y_{i}\,log(\hat{p}) + (-1+y_{i}) \, log(1-\hat{p})\\ 
$$

<br/>

지금까지 Log loss 함수는 이진 분류 모델이 예측한 확률값의 negative log likelihood를 계산한 결과임을 확인하였습니다.  식의 의미를 되짚어 보겠습니다.  Log의 밑이 자연상수일 때 정보 이론[^1]의 렌즈로  negative log likelihood를 바라보겠습니다. 모집단의 세계에서 알 수 있는 반응 변수에 대한 모분포(R) 정보를 현실에 주어진 데이터를 재료로 삼아 반응 변수를 추정한 분포(I) 정보를 통해 전달할 때 필요한 추가적인  정보량(비트 단위)일 것입니다. 

<br/>

### Brier Score
___________________

- Brier score[^2]는 예측된 확률값들과 기댓값의 MSE(Mean Squared Error)를 계산한 지표입니다. 
- Threshold Metric중 Recall-Precision curve처럼 양성 클래스의 확률값에 초점을 맞춥니다.
- 다만 Recall-Precision curve와는 달리 다중 클래스에도 적용할 수 있습니다. 

<br/>

$$
BS(Brier Score) = \frac{1}{N} \times \sum_{i=1}^{N}(\hat{y_{i}} - y_{i})^2
$$

<br/>

확률의 첫번째 공리와 MSE 수식 자체의 특성이 결합되어 Brier Score는 굉장히 작은 값이 나올 가능성이 높습니다. 이런 문제점을 해결하기 위해서 Brier score를 베이스 라인 모델의 참고 점수(Reference score)를 사용하여 변형합니다. 
 
<br/>

$$
BrierSkillScore = 1 - \frac{Brier \, score}{Brier \, score[reference]} 
$$

<br/>


가상의 불균형 데이터를 만들어서 Brier Score를 살펴보겠습니다. 먼저 10000개 샘플에 관하여 두 가지 예측 변수들과 클래스가 0이거나 1인 클래스를 구성하겠습니다. 노이즈 데이터의 비율은 실험의 명확성을 위해 0으로 설정하였고 다수 클래스에 대한 가중치는 90%로 설정하였습니다. 그리고 y 클래스의 비율로 Stratified random sampling을 거치겠습니다.

```python
X, y = make_classification(n_samples=10000,
          weights=[0.90],
          n_features = 2,
          n_redundant=0,
          n_clusters_per_class=1,
          random_state = 33,
          flip_y=0)

unique_value, counts = np.unique(y, return_counts=True)
list_y = list(zip(unique_value, counts))
{i[0] : i[1] for i in list_y}    
```
```
{0: 9001, 1: 999}
```

모든 케이스를 'Negative'로 예측할 경우 Brier Score는 다음과 같습니다.


```python
all_negative_probs = [0.0 for _ in range(len(testy))]
brier_score = np.mean((testy - all_negative_probs)**2)
print('P(-) brier_score = ', brier_score)   
```

```
P(-) brier_score =  0.1
```

반면에 모든 케이스를 'Positive'로 예측할 경우 Brier Score는 다음과 같습니다.

```python
all_positive_probs = [1.0 for _ in range(len(testy))]
brier_score = np.mean((testy - all_positive_probs)**2)
print('P(+) brier_score = ', brier_score)  
```

```
P(+) brier_score =  0.9
```

<br/>

다음으로 Brier Skill Score를 살펴보겠습니다. 모든 확률이 0.1일 때를 가정한 값을 Reference Score로 설정하겠습니다. 그리고 모든 케이스를 '+'로 예측한 경우, 모든 케이스를 '-'로 예측한 경우, 그리고 베이스라인 모델에 사용했던 0.1의 확률을 사용하여 BSS를 개별적으로 구하면 다음과 같습니다.

```python
def refer_bss(prob_constant,ref_constant,y):
    case_probs = [prob_constant for _ in range(len(y))]
    ref_probs = [ref_constant for _ in range(len(y))]
    case_bs = np.mean((testy-case_probs)**2)
    ref_bs = np.mean((testy-ref_probs)**2)
    return 1 - (case_bs/ref_bs)
    
print('Rererence brier skill score = ', round(refer_bss(0.1,0.1,testy),4))
print('P(-) brier skill score = ', round(refer_bss(0.0,0.1,testy),4))
print('P(+) brier skill score = ', round(refer_bss(1.0,0.1,testy),4))
```

```
Rererence brier skill score =  0.0
P(-) brier skill score =  -0.1111
P(+) brier skill score =  -9.0
```

<br/>


## Probability Calibration

반응 변수의 분포가 심하게 왜곡된 경우 모델이 예측한 모델은 다수 클래스를 지나치게 선호하게 됩니다. 이런 경향을 완화하기 위해서 예측된 확률을 대상으로 Calibration을 적용합니다. 교정된 확률들은 실제 사건의 가능도를 반영합니다. 

안타깝게도 Calibration이 적용된 모델은 아주 많지 않습니다. 대표적으로 MLE 추정을 활용한 Logistic Regression이 있습니다. Logistic regression의 cost function은 위에서 유도한 Log loss function을 사용하며 식은 다음과 같습니다[단, \\( N_{x} = m\\), \\(\hat{p} = h_\theta\\)]

<br/>

$$
\begin{aligned}
J(\theta)  = 
-\frac{1}{m}\sum_{i=1}^m 
\left[ y^{(i)}\log\left(h_\theta \left(x^{(i)}\right)\right) +
(1 -y^{(i)})\log\left(1-h_\theta \left(x^{(i)}\right)\right)\right] 
\end{aligned}
$$

$$
where \, h_{\theta}(x^{(i)}) = \frac{1}{1+\exp[-\alpha -\sum_j \theta_j x^{(i)}_j]}
$$

<br/>

Cost function을 최소화 하기 위해서 파라미터인 \\( \theta_{j}\\\)로 편미분[^3]한 결과는 0입니다.

<br/>

$$
\begin{aligned}
\small
\frac{\partial J(\theta)}{\partial \theta_j}  = 
\frac{\partial}{\partial \theta_j} \,\frac{-1}{m}\sum_{i=1}^m 
\left[ y^{(i)}\log\left(h_\theta \left(x^{(i)}\right)\right) +
(1 -y^{(i)})\log\left(1-h_\theta \left(x^{(i)}\right)\right)\right]
\\[2ex]\small\underset{\text{linearity}}= \,\frac{-1}{m}\,\sum_{i=1}^m 
\left[ 
y^{(i)}\frac{\partial}{\partial \theta_j}\log\left(h_\theta \left(x^{(i)}\right)\right) +
(1 -y^{(i)})\frac{\partial}{\partial \theta_j}\log\left(1-h_\theta \left(x^{(i)}\right)\right)
\right]
\\[2ex]\Tiny\underset{\text{chain rule}}= \,\frac{-1}{m}\,\sum_{i=1}^m 
\left[ 
y^{(i)}\frac{\frac{\partial}{\partial \theta_j}h_\theta \left(x^{(i)}\right)}{h_\theta\left(x^{(i)}\right)} +
(1 -y^{(i)})\frac{\frac{\partial}{\partial \theta_j}\left(1-h_\theta \left(x^{(i)}\right)\right)}{1-h_\theta\left(x^{(i)}\right)}
\right]
\\[2ex]\small\underset{h_\theta(x)=\sigma\left(\theta^\top x\right)}=\,\frac{-1}{m}\,\sum_{i=1}^m 
\left[ 
y^{(i)}\frac{\frac{\partial}{\partial \theta_j}\sigma\left(\theta^\top x^{(i)}\right)}{h_\theta\left(x^{(i)}\right)} +
(1 -y^{(i)})\frac{\frac{\partial}{\partial \theta_j}\left(1-\sigma\left(\theta^\top x^{(i)}\right)\right)}{1-h_\theta\left(x^{(i)}\right)}
\right]
\\[2ex]\Tiny\underset{\sigma'}=\frac{-1}{m}\,\sum_{i=1}^m 
\left[ y^{(i)}\,
\frac{\sigma\left(\theta^\top x^{(i)}\right)\left(1-\sigma\left(\theta^\top x^{(i)}\right)\right)\frac{\partial}{\partial \theta_j}\left(\theta^\top x^{(i)}\right)}{h_\theta\left(x^{(i)}\right)} -
(1 -y^{(i)})\,\frac{\sigma\left(\theta^\top x^{(i)}\right)\left(1-\sigma\left(\theta^\top x^{(i)}\right)\right)\frac{\partial}{\partial \theta_j}\left(\theta^\top x^{(i)}\right)}{1-h_\theta\left(x^{(i)}\right)}
\right]
\\[2ex]\small\underset{\sigma\left(\theta^\top x\right)=h_\theta(x)}= \,\frac{-1}{m}\,\sum_{i=1}^m 
\left[ 
y^{(i)}\frac{h_\theta\left( x^{(i)}\right)\left(1-h_\theta\left( x^{(i)}\right)\right)\frac{\partial}{\partial \theta_j}\left(\theta^\top x^{(i)}\right)}{h_\theta\left(x^{(i)}\right)} -
(1 -y^{(i)})\frac{h_\theta\left( x^{(i)}\right)\left(1-h_\theta\left(x^{(i)}\right)\right)\frac{\partial}{\partial \theta_j}\left( \theta^\top x^{(i)}\right)}{1-h_\theta\left(x^{(i)}\right)}
\right]
\\[2ex]\small\underset{\frac{\partial}{\partial \theta_j}\left(\theta^\top x^{(i)}\right)=x_j^{(i)}}=\,\frac{-1}{m}\,\sum_{i=1}^m \left[y^{(i)}\left(1-h_\theta\left(x^{(i)}\right)\right)x_j^{(i)}-
\left(1-y^{i}\right)\,h_\theta\left(x^{(i)}\right)x_j^{(i)}
\right]
\\[2ex]\small\underset{\text{distribute}}=\,\frac{-1}{m}\,\sum_{i=1}^m \left[y^{i}-y^{i}h_\theta\left(x^{(i)}\right)-
h_\theta\left(x^{(i)}\right)+y^{(i)}h_\theta\left(x^{(i)}\right)
\right]\,x_j^{(i)}
\\[2ex]\small\underset{\text{cancel}}=\,\frac{-1}{m}\,\sum_{i=1}^m \left[y^{(i)}-h_\theta\left(x^{(i)}\right)\right]\,x_j^{(i)} \\[2ex]\small=\frac{1}{m}\sum_{i=1}^m\left[h_\theta\left(x^{(i)}\right)-y^{(i)}\right]\,x_j^{(i)} \\[2ex]
\small= 0
\end{aligned}
$$

식을 조금 정리해 주면 다음과 같이 재밌는 결과가 나옵니다.

<br/>

$$
\sum_{i=1}^m h_\theta\left(x^{(i)}\right)x_j^{(i)}=\sum_{i=1}^m y^{(i)}\,x_j^{(i)}
$$


<br/>

이러한 성질을 담보하지 않는 Tree 계열 모델들과 SVM은 교정된 확률을 가지지 못하기 때문에 이러한 모델의 예측값을 확률로 설정할 때는 반드시 Calibration을 인위적으로 적용해야 합니다.


<br/>

그래서 앞의 수식처럼 학습 데이터에서 관측된 분포와 모델의 예측 확률 값들의 집합이 잘 일치하도록 스케일링해야 합니다. 예측 확률들을 스케일링하는 방법은 크게 두 가지 입니다.

- Platt Scaling[^4]
	- Platt Scaling은 SVM의 예측 확률값들을 스케일링하기 위해 개발되었습니다. Logistic regression 모델에서 적용된 기법이 사용됩니다. 	
- Isotonic Regression[^5]
	-  가중치가 사용된 최소 제곱 회귀 모델입니다. 
	-  Platt Scaling에 비해 많은 데이터가 필요하지만 보다 더 일반화된 모델 성능을 얻을 수 있습니다.

<br/>

## Reference

[^1]: https://en.wikipedia.org/wiki/Information_theory
[^2]: https://en.wikipedia.org/wiki/Brier_score
[^3]: https://stats.stackexchange.com/questions/278771/how-is-the-cost-function-from-logistic-regression-derivated/278812#278812
[^4]: https://en.wikipedia.org/wiki/Platt_scaling
[^5]: https://en.wikipedia.org/wiki/Isotonic_regression






