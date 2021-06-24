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
	

Transfer Windows — Predicting English Premier League club ratings

Photo by Tim Bechervaise on Unsplash
When Leicester City won the 2015–2016 English Premier League title, they completed a 5000–1 miracle season and qualified for the UEFA Champions League for the first time in their history. The list of teams that have won a Premier League title is pretty short — every single Premier League season has been won by either Manchester United (13), Chelsea (5), Manchester City (5), Arsenal (3), Liverpool (1), Leicester City (1) or Blackburn Rovers (1).
Even the most casual of soccer fans will look at that list and recognize that the first five teams on that list are some of the most recognizable club brands in all of sports. Manchester United, Manchester City, Chelsea, Arsenal and Liverpool are five of the Big Six clubs (which includes Tottenham), so-called because of their global brand positioning and their relative dominance over the rest of the Premier League field. What sets those Big Six clubs apart from nearly every other club Premier League team? Money, and specifically, the market value of the players they bring in. From transfermarkt, which tracks club and player market values, the current average value for the Big Six clubs is €803.23 million, while the average for the remaining 14 clubs is only €269.48 million. It’s the money behind the larger clubs that makes them so dominant. Money buys players and managers, players and managers win more trophies, more trophies improves brand positioning, and the clubs remain at the top of the league.
Knowing what we know about money in soccer (namely, that it is king), but simply having a larger war chest does not equate to easy success. Tottenham is one of the Big Six clubs, but have yet to win a Premier League title. Arsenal is one of the Big Six clubs, but have finished below 5th place, and as low as 8th, in each of the last five seasons.
The question then becomes: how does the way that clubs spend money impact their performance? Clubs spend and earn during two transfer windows each season to build the best squad for the best value, and those players ultimately drive the performance of each club. To get to the bottom of how money impacts a club’s strength, I decided to look at the impact that transfers have on club’s strength, indicated by something called the SPI rating.
The SPI rating was created by the 
FiveThirtyEight
 team to predict the outcomes of matches and seasons. They have expanded their leagues over the years and now have a full ranking of 600+ men’s professional club teams. The top-ranked team as of this post? Manchester City, perennial Premier League powerhouse and financial mega-team. The worst-ranked team? Scunthorpe, which just secured its spot in the English League Two after a relegation battle.
Using transfer data for the seasons 2016–2020 and SPI ratings for each club in the Premier League during those seasons, I used Bayesian Ridge regression to assess how clubs spend their money impacts their SPI rating.
One quick note before continuing: This is not a step-by-step tutorial for tuning algorithm parameters or feature selection. It’s also not an in-depth explanation of soccer analytics. This is a first project, involving data in a sport that I’m passionate about. For me, it’s a way to resolve my first project and learn for the future. For others, I hope it’s a good look at what a beginner’s project looks like, and for everyone, I hope it motivates you to undertake a project that seems intimidating. Keep reading on!
The Data & Feature Creation
I used two data sets to examine transfers and SPI ratings.
For transfers, I used data scraped from transfermarkt that 
Ewen
 has made available in a Github repository. There is an incredible amount of data, and I encourage anyone interested in market values or transfer activity to have a browse through the work he’s done. For SPI ratings, I used the 
FiveThirtyEight
 repository that includes data on soccer matches in professional soccer leagues from across the globe.
At the beginning of the project, I planned to use both the Championship and the Premier League divisions of the English football pyramid. As this is my first ever modelling exercise (I’m new at this), I decided to limit the data to just the Premier League. I felt most comfortable working with each data set individually and then merging for some final clean up; now that I have more confidence in my abilities using notebooks and the pandas library, I’d probably approach the project differently if I try it again.
I used the SPI ratings for each team, home and away, across each of the five seasons I looked at and created a new feature with the average SPI rating per season.


The violin plot shows the wide spread of SPI ratings in the Premier League. Interestingly, the minimum rating increased from 2017 to 2018, as weaker teams were relegated out, and stronger teams were promoted in.
Individual clubs also have a spread of SPI ratings across their seasons. The gray bar assigned to Leicester City illustrates much better or worse a team performed across the five seasons, on average. The six bars that have distributions farthest to the right? Those are the Big Six. Money matters.

From the correlation map above, it is clear that the features most closely correlated with the Average SPI score (that weren’t used in its creation!) are the Max Fee Spent, the Average Fee spent in a transfer window, and the Total Earned by a club during a transfer window, on fees gained when selling players to clubs. Interestingly, the total transfers in and out of a club in a given season were negatively correlated with the Average SPI rating. We are starting to see evidence of ‘quanlity over quantity’ at work in the Premier League. Quick side note: I removed the average home and away SPI scores from my analysis, since they were used to create the Average Season SPI Score feature.
Modelling & Analysis
As I mentioned above, this is my first project, so I tried out as many different methods as I could to scale and model the data.

I used MinMax, RobustScaler and StandardScaler to scale the data before splitting into test sets. I settled on using the MinMax scaler in the project. I split the data into training and test sets, and then tried my hand at using a variety of regression algorithms. My first attempt, which I won’t show here but that you can see in the repository, had some really high scores! My first project, my first try, massive scores!
Then I realized that I was still had Average Home and Average Away SPI ratings in my data. Since they were used to create the Average Season SPI score, I had inflated scores. I tried again, dropping the home and away ratings.
You can see the results of my second attempt below. The highest scoring algorithm — Bayesian Ridge Regressor.
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
I then looked at the feature importances for the Bayesian Ridge Regressor to determine which features were most predictive of a club’s average SPI score in a given season.

Feature Importances — Bayesian Ridge Regressor
Voila! The most predictive features are:
Max Fee Spent: the maximum transfer fee spent on a single player
Total Spent (In): the total spent by a club to bring players in
Total Earned (Out): the total earned by a club when selling players
Conclusions & Next Steps
Money is king. That hasn’t, and won’t, change in professional soccer leagues, and teams must manage their finances and team goals accordingly. Clubs at the top, usually both in performance and in war chest, are spending vast sums on players in order to maximize their position at the top of the table and qualify for European competitions. Clubs farther down the table have more modest, but equally important, goals: improve their position in the table each season, stay out of a relegation fight, manage expenses to keep the club in a good financial position. Given that, no matter how much money a club has, we now know that there are steps every team can take in a transfer window to improve their SPI ranking, which will likely result in a higher position on the table at the end of each season.
First, clubs should target one key player during each window and put a big sum behind their efforts. Spending more on one player is more impactful than spreading the fees out across lots of positions.
Second, clubs should manage expenses to maximize the money they can put toward bringing players in. A few seasons ago, Tottenham brought in zero players during a transfer window. They are still feeling the result of that tactic and have not seen the results in the Premier League table that they want.
Third, clubs should make good bets on players and sell them of to ‘balance the checkbook’ each transfer window. Using development academies and off-loading quality players earlier rather than later will help clubs manage their fees in and out. After the 2020 — 2021season, Liverpool are shopping one of their top defensive players, because his value is inflated off last season’s performance. He’s in demand across Europe. Continuing to include him in the squad is too great a risk to his market value. Off-loading him now, even though he is a quality player with a bright future, will help them maintain some balance in the next transfer window.
As I’ve mentioned a couple of times, this was my first ‘data science’-related project. I am no data scientist, and I am still learning the math behind the Bayesian Ridge Regressor, decision trees and other algorithms. I’m happy with the results, though I know there are some next steps to explore when I look back at this project again:
First, I need to research why Bayesian Ridge performed better than decision trees. Understanding why an algorithm is outperforming others, even just slightly, is important. Knowing the code isn’t enough, in my opinion; it’s a good start though.
Second, speaking of performance, I need to go back to my pre-processing and modelling and tune the algorithm parameters to try to improve my scores. I have a second project that I’m finishing now that explores employee retention, and I’ve learned a lot about parameter tuning. I’ll come back to this soccer transfers project and apply my knowledge to see what I can do to improve model performance.
Third, finishing projects is a real confidence booster for me! I took long breaks away from this, partially due to life events, but mostly due to being nervous about my abilities. It was hard to brush aside those negative thoughts: “You aren’t good at this. You don’t know enough. This is a joke.” What I’ve learned is that getting to a stopping point is a huge confidence boost. For future projects, I know I can keep my mind focused on getting to a stopping point, evaluating my efforts to that point, and deciding on next steps.
And as my mentor said when we were discussing my confidence — “If think you don’t know enough, go learn some stuff!” Thanks, 
Rafael Castillo Alcibar
.
