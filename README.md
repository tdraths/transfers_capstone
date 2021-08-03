# Using Transfer Data to Predict English Premier League team strength
Repository holding all work for Springboard Capstone 1: Impact of Transfers on English Football club strength

### **UPDATE**: I revisited this project to develop a more complete project and improve the model. You can see the [repository here](https://github.com/tdraths/spi_transfers_global)

[Medium Post](https://tdraths.medium.com/transfer-windows-predicting-english-premier-league-club-ratings-779b37008353)

![Cover](https://miro.medium.com/max/9650/0*wPuBg7ZwWs8Fpg4V)


This is my first 'data science' project. I'll revisit it again after completing my first attempts using machine learning algorithms, to apply any lessons learned.

I sourced data from: 
https://github.com/ewenme/transfers
https://github.com/fivethirtyeight/data/tree/master/soccer-spi

**Background:**
In soccer, money is king. The top clubs in each league tend to be the wealthiest or most valuable clubs, and their large war chests enable them to spend where it matters - transfer fees for top quality players. This project explores how transfer spends impacts club strength, measured by the SPI rating developed by FiveThirtyEight.

Problem Questions: How do club transfer windows affect team strength, when considering the sums spent each window? What can clubs with less available money do to increase their odds of a better performance each season, as it relates to SPI rating?

### Data Sourcing & Cleaning
I sourced data on transfer activity amongst English Premier League teams, and SPI ratings. Links to the data are above. I had to do some decent cleaning and merging of data sets in order to get one workable data set. The process by which I cleaned and prepared the data can be found [here](https://github.com/tdraths/transfers_capstone/blob/main/notebooks/data_cleaning_transfers_data.ipynb) (transfers) and [here](https://github.com/tdraths/transfers_capstone/blob/main/notebooks/data_cleaining_spi_data.ipynb) (SPI ratings).

A few challenges I encountered cleaning the data.
 - The data sets used different names to identify EPL teams. I had to develop a standardization for naming and apply it across the two data sets before merging, to make the merged / final set easier to manage.
 - There are were a number of duplicate records that involved loaned players, who are traded from one team to another for a portion or all of a given season and then returned to their 'home team' at the end of the loan period. Often, players are loaned to and from the same clubs twice in a season, and those duplicate records were dropped as they were throwing off the data.
 - Limiting the data to the EPL, for five seasons, presented me with only 100 final data points from which to build a model. Not a lot, and I'd include more leagues in future analyses.

### Exploratory Analysis & Creating Features
[Notebook](https://github.com/tdraths/transfers_capstone/blob/main/notebooks/New_Features.ipynb)
This is one area where I probably could have done more work, and if I revisit this project in the future, I think I'll look deeper at variable interactions. I did improve my plotting skills, focusing on violin plots and box plots to show the distributions of average transfer fees, average SPI ratings, and total tranfers per season. I also show variable correlations with a correlation map.

#### Average SPI Scores
![Average SPI Scores by Season - Violin Plot](https://github.com/tdraths/transfers_capstone/blob/main/figures/download.png)

![Average SPI Scores by Season - Box Plot](https://github.com/tdraths/transfers_capstone/blob/main/figures/download%20(1).png)

#### Average Fees Spent
![Average Fee Spent by Season - Box Plot](https://github.com/tdraths/transfers_capstone/blob/main/figures/download%20(2).png)

One area of improvement I feel really good about: I thought clearly about the features I was interested in seeing, and used good code to help me develop them for the data set. This includes the average SPI rating using the home and away SPI ratings, as well as average fees spent per position group (e.g. Goalkeeper, Striker). My original data sets have no real common data points - they are largely just records of matches and transfers between clubs. To develop features that could be used to predict SPI rating based on transfer activity, I created 22 additional features.

#### Features Correlation
![Features Correlation Map](https://github.com/tdraths/transfers_capstone/blob/main/figures/download%20(3).png)

### Pre-Processing & Modelling
[Notebook](https://github.com/tdraths/transfers_capstone/blob/main/notebooks/Pre-Processing%20and%20Modelling.ipynb)
I used MinMax, StandardScaler and RobustScaler to illustrate a scaled dataset before splitting my data into training and testing sets. I settled on using the MinMax scaler in pre-processing.

#### Scaled Data Illustration
![Scaled Data using different scaling techniques](https://github.com/tdraths/transfers_capstone/blob/main/figures/download%20(4).png)

I tried a wide range of algorithms to find the most predictive model. During my first attempt, I (accidentally) included the Home and Away Average SPI scores, which were used to calculate the Average Season SPI Score. This resulted in very high / good scores for nearly every algorithm, and one of my first big lessons in building models - be careful which features you are including!

I tried again, without the Home & Away scores, and while my scores dropped considerably, at least the results are good results. You can see the results of that effort below:

Model Name | R<sup>2</sup> Score | MAE
---------- | ------------------- | ---
OLS Linear Regression | 0.1268 | 4.8705
NNLS Linear Regression | 0.3194 | 4.5814
Least-Angle Regression | 0.3541 | 6.5048
Bayesian Ridge Regression | 0.4189 | 5.4320
Support Vector Regression | 0.1083 | 8.9043
Gaussian Process Regression | 0.2136 | 6.3722
Partial Least Squares Regression | 0.4004 | 4.8793
Decision Tree - Default Max Depth | -0.2541 | 6.0600
Decision Tree - Max Depth 2 | 0.0834 | 5.134
Decision Tree - Max Depth 5 | -0.0540 | 7.3408
Decision Tree - Max Depth 8 | -0.0181 | 6.7400
ADABoost Tree - Default Max Depth | 0.3100 | 4.6300
ADABoost Tree - Max Depth 2 | 0.2400 | 4.7757
ADABoost Tree - Max Depth 5 | 0.2994 | 4.8814
ADABoost Tree - Max Depth 8 | 0.3153 | 4.7050

#### Choosing an Algorithm & Feature Importances
I selected Bayesian Ridge regression to continue my analysis, because it had the highest R<sup>2</sup> Score. In this project, I did not select out features or tune parameters to improve the model score. When I explore this dataset again in the future, I will use what I've learned **since** completing this project to develop a better model and improve predicitability scores. For my purposes on this project, I looked at the feature importances to assess which features were most predictive of an SPI score, and developed conclusions from there.

![Feature Importances - Baysesian Ridge](https://github.com/tdraths/transfers_capstone/blob/main/figures/download%20(5).png)

### Conclusions & Next Steps
Money is king. That hasn’t, and won’t, change in professional soccer leagues, and teams must manage their finances and team goals accordingly. Clubs at the top, usually both in performance and in war chest, are spending vast sums on players in order to maximize their position at the top of the table and qualify for European competitions. Clubs farther down the table have more modest, but equally important, goals: improve their position in the table each season, stay out of a relegation fight, manage expenses to keep the club in a good financial position. Given that, no matter how much money a club has, we now know that there are steps every team can take in a transfer window to improve their SPI ranking, which will likely result in a higher position on the table at the end of each season.

 - **First**, clubs should target one key player during each window and put a big sum behind their efforts. Spending more on one player is more impactful than spreading the fees out across lots of positions.
 - **Second**, clubs should manage expenses to maximize the money they can put toward bringing players in. A few seasons ago, Tottenham brought in zero players during a transfer window. They are still feeling the result of that tactic and have not seen the results in the Premier League table that they want.
 - **Third**, clubs should make good bets on players and sell them of to ‘balance the checkbook’ each transfer window. Using development academies and off-loading quality players earlier rather than later will help clubs manage their fees in and out. After the 2020 — 2021season, Liverpool are shopping one of their top defensive players, because his value is inflated off last season’s performance. He’s in demand across Europe. Continuing to include him in the squad is too great a risk to his market value. Off-loading him now, even though he is a quality player with a bright future, will help them maintain some balance in the next transfer window.

As I’ve mentioned a couple of times, this was my first ‘data science’-related project. I am no data scientist, and I am still learning the math behind the Bayesian Ridge Regressor, decision trees and other algorithms. I’m happy with the results, though I know there are some next steps to explore when I look back at this project again:

 - I need to research why Bayesian Ridge performed better than decision trees. Understanding why an algorithm is outperforming others, even just slightly, is important. Knowing the code isn’t enough, in my opinion; it’s a good start though.
 - Speaking of performance, I need to go back to my pre-processing and modelling and tune the algorithm parameters to try to improve my scores. I have a second project that I’m finishing now that explores employee retention, and I’ve learned a lot about parameter tuning. I’ll come back to this soccer transfers project and apply my knowledge to see what I can do to improve model performance.
 - Finishing projects is a real confidence booster for me! I took long breaks away from this, partially due to life events, but mostly due to being nervous about my abilities. It was hard to brush aside those negative thoughts: “You aren’t good at this. You don’t know enough. This is a joke.” What I’ve learned is that getting to a stopping point is a huge confidence boost. For future projects, I know I can keep my mind focused on getting to a stopping point, evaluating my efforts to that point, and deciding on next steps.

And as my mentor said when we were discussing my confidence — “If think you don’t know enough, go learn some stuff!” Thanks, 
Rafael Castillo Alcibar


.







