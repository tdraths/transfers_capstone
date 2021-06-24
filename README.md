# Transfers Capstone
Repository holding all work for Springboard Capstone 1: Impact of Transfers on English Football club strength

This is my first 'data science' project. I'll revisit it again after completing my first attempts using machine learning algorithms, to apply any lessons learned.

I sourced data from: 
https://github.com/ewenme/transfers
https://github.com/fivethirtyeight/data/tree/master/soccer-spi

My algorithm metrics:

OLS Linear Regression 
	Season model score: 0.1268
	Season MAE: 4.8705
	SPI model score: -0.541
	SPI MAE: 3.58
  
Non-negative Least Squares Linear Regression 
	Season model score: 0.3194
	Season MAE: 4.5814
	SPI model score: -0.5924
	SPI MAE: 2.6768
  
Least-Angle Regressor 
	Season model score: 0.3541
	Season MAE: 6.5048
	SPI model score: -0.1531
	SPI MAE: 2.2011
  
Bayesian Ridge 
	Season model score: 0.4189
	Season MAE: 5.432
	SPI model score: -0.1555
  
Suppor Vector 
	Season model score: 0.1083
	Season MAE: 8.9043
	SPI model score: -0.1068
	SPI MAE: 2.259
  
Gaussian Process Regressor 
	Season model score: 0.2136
	Season MAE: 6.3722
	SPI model score: -0.587
	SPI MAE: 1.7552
  
Partial Least Squares Regressor 
	Season model score: 0.4004
	Season MAE: 4.8793
	SPI model score: -0.6122
	SPI MAE: 3.2796
  
DecisionTree Models with Default Max Depth 
	Season model score: 0.2041
	Season MAE: 5.595
	SPI model score: -0.4527
	SPI MAE: 3.4612

DecisionTree Models with Max Depth of 2 
	Season model score: 0.0834
	Season MAE: 5.134
	SPI model score: -0.3567
	SPI MAE: 2.7878

DecisionTree Models with Max Depth of 5 
	Season model score: 0.022
	Season MAE: 7.3408
	SPI model score: -0.5005
	SPI MAE: 3.9308

DecisionTree Models with Max Depth of 8 
	Season model score: 0.1159
	Season MAE: 5.675
	SPI model score: -0.7262
	SPI MAE: 2.785

ADABoosted - DecisionTree Models with Default Max Depth 
	Season model score: 0.31
	Season MAE: 4.63
	SPI model score: -0.1058
	SPI MAE: 2.7712

ADABoosted - DecisionTree Models with Max Depth of 2 
	Season model score: 0.24
	Season MAE: 4.7757
	SPI model score: -0.4566
	SPI MAE: 2.885

ADABoosted - DecisionTree Models with Max Depth of 5 
	Season model score: 0.2994
	Season MAE: 4.8814
	SPI model score: -0.139
	SPI MAE: 2.7266

ADABoosted - DecisionTree Models with Max Depth of 8 
	Season model score: 0.3153
	Season MAE: 4.705
	SPI model score: -0.1174
	SPI MAE: 2.7475
