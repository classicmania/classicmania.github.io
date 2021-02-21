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
 
  <br/>
 
| |Computation Cost|공통의 키|ROW 처리 방법|
|:------:|:---:|:---:|:---:|
|Inner Join|△|<span style="color:red">O</span>|Left dataset key and Right dataset key|
|Outer Join|△|<span style="color:red">O</span>|Left dataset key or Right dataset key|
|Left Outer Join|△|<span style="color:red">O</span>|Left dataset Key only|
|Right Outer Join|△|<span style="color:red">O</span>|Right dataset key only|
|Left Semi Join|△|<span style="color:red">O</span>|Left data set key matching Right dataset key|
|Left Anti Join|△|<span style="color:red">O</span>|Left data set key not maching Right dataset key|
|Natural Join|↑|<span style="color:red">X</span>|Implicit join|
|Cross Join|↑|<span style="color:red">X</span>|Catesian Join|

 <br/>
 
 조인할 때 스파크는 셔플 조인(Shuffle Join)과 브로드캐스트 조인(Broadcast Join)방식 중 하나의 통신 방식을 사용합니다. 
 - Shuffle Join
	 - Shuffle Join은 전체의 노드간의 통신을 발생 시킵니다. 
	 - 개별 데이터프레임의 파티션에서 Data Skewness가 발생할 경우 Shuffle Join의 Cost는 기하급수적으로 늘어날 것입니다. 
	 - 일반적으로 Join의 대상이 되는 테이블이 클 경우 Shuffle Join이 활용됩니다.
- Broadcast Join
	- Shuffle Join과 다르게 Broadcast Join은 Join의 대상이 되는 테이블의 크기 차이가 클 경우 적용됩니다.
	- 만약 Low Level이 아닌 High Level API를 활용하면 Optimizer에서 Broadcast join을 사용하도록 힌트를 줄 수 있지만 일반적으로 스파크 자체내에서 자동으로 Broadcast 방식으로 통신하는 것을 쉽게 알 수 있습니다.
 
 <br/>
 
![Broadcast_join_spark](/assets/img/post_img/Broadcast_join_spark.jpeg)_Broadcast_hash _join_



 
