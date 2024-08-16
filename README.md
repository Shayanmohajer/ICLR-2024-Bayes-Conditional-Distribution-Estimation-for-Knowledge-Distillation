# Bayes Conditional Distribution Estimation for Knowledge Distillation Based on Conditional Mutual Information (ICLR-2024)

## Abstract 
It is believed that in knowledge distillation (KD), the role of the teacher is to provide
an estimate for the unknown Bayes conditional probability distribution (BCPD) to
be used in the student training process. Conventionally, this estimate is obtained by
training the teacher using maximum log-likelihood (MLL) method. To improve
this estimate for KD, in this paper we introduce the concept of conditional mutual
information (CMI) into the estimation of BCPD and propose a novel estimator
called the maximum CMI (MCMI) method. Specifically, in MCMI estimation, both
the log-likelihood and CMI of the teacher are simultaneously maximized when the
teacher is trained. Through Eigen-CAM, it is further shown that maximizing the
teacher’s CMI value allows the teacher to capture more contextual information in
an image cluster. Via conducting a thorough set of experiments, we show that by
employing a teacher trained via MCMI estimation rather than one trained via MLL
estimation in various state-of-the-art KD frameworks, the student’s classification
accuracy consistently increases, with the gain of up to 3.32%. This suggests that
the teacher’s BCPD estimate provided by MCMI method is more accurate than
that provided by MLL method. In addition, we show that such improvements in
the student’s accuracy are more drastic in zero-shot and few-shot settings. Notably,
the student’s accuracy increases with the gain of up to 5.72% when 5% of the
training samples are available to the student (few-shot), and increases from 0%
to as high as 84% for an omitted class (zero-shot).



## Requirements
- python 3.8
- pytorch 1.11.0
- CUDA 11.3.1

## Clone the repository
git clone https://github.com/Shayanmohajer/ICLR-2024-Bayes-Conditional-Distribution-Estimation-for-Knowledge-Distillation.git


## Results
![Screenshot of a comment on a GitHub issue showing an image, added in the Markdown, of an Octocat smiling and raising a tentacle.](https://drive.google.com/file/d/1qYNDg85AGMRTsy87BOr6DfBIIFndn4hz/view?usp=sharing)
