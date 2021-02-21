---
title: Spark Tuning과 관련된 몇 가지 발견들
author: Kwon Suncheol
categories: [Spark]
tags: [Spark,RDD,Negative_Sampling,Data_sampling,Spark_Tuning,RFM]
pin: false
math: true
comments: true
---

![spark](/assets/img/post_img/Apache_spark.png)

<br/>

 최근에 고객 세분화(Customer Segmentation) RFM 모델 디버깅 작업과 팀원 중 한분께서 Session Based RS를 고안하실 때 필요한 순차 데이터(Sequential Data) 전처리 작업을 스파크로 진행하면서 경험하였던 것들 중 몇 가지를 공유하고자 합니다. 
 
<br>

## Spark Tuning
___________________


### 몇 가지 전제들

 'Data Skewness in spark'를 전개 하기 위해서 필요한 몇 가지 점들을 공유하겠습니다. <br/>
 
 스파크는 기본적으로 다음 이미지와 같이 동작합니다. Local Mode도 지원하지만 실무에서는 Cluster Mode를 일반적으로 활용합니다. 스파크 애플리케이션의 개수가 RFM 뿐만 아니라 다른 어플리케이션도 존재하므로 클러스터 매니저는 YARN으로 설정하고 처리해야 하는 데이터의 기간이 길고 크므로 Driver Memory와 Executor Memory는 충분히 지정하였습니다.
 
 <br/>
 
![spark_cluster_structure](/assets/img/post_img/Spark_Structure_cluster_mode.png)
 
 <br/>
 
 다음으로 스파크가 제공하는 Join에 대해 살펴보겠습니다. 스파크는 아래의 표와 같이 다양한 조인 표현식(Join Expression)을 제공합니다. 
 
