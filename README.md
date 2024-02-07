# Predicting the Pound (Master's Level Dissertation Project)
Predicting the Pound is the project I undertook as part of my dissertation module whilst studying for my MSc Finance degree at Newcastle University Business School. The focus of the project was to investigate the performance of various volatility models, specifically GARCH-type and Implied Volatility, when attempting to forecast the volatility of the GBP/USD and EUR/GBP currency pairs. This repository contains the various Python scripts used to evaluate the models and my final submitted [dissertation paper](https://github.com/boyla950/predicting-the-pound/blob/main/paper.pdf).

## The Dataset
In this project, I downloaded daily price data for each of the currency pairs from Yahoo Finance from a period ranging between June 2018 to June 2023. The raw data contained the open, close, high and close prices for each currency pair as well as the Adjusted Close prices and the daily trading volume. This data was then cleaned to focus solely on the close prices each day.

As volatility is not directly observable, it was necessary to select an appropriate proxy for its value. While there is much discourse in the academic literature as to the most appropriate proxy for this, we opt to use the 20-day standard deviation of returns. This results in much less noisy data than the absolute values of daily returns (another potential proxy), making it much more useful for forecasting.


## The Models
Whilst the focus of the project was on the performance of GARCH-type and Implied Volatility models, an Exponentially Weighted Moving Average model (EWMA) was applied as a baseline. Of the various GARCH-type models available, GARCH, EGARCH, GJR-GARCH and TGARCH were used in this projects experiments. In order to select the best parameter sets for each model, we iterated through all possible combinations of *p* and *q* between 1 and 5, selecting up to parameter sets for each model by selecting the sets with the best AIC and BIC scores.

For the implied volatility models, a two simple OLS regression models were used. The first had only one variable; the implied volatility value of the previous day. The second had two variables; the implied volatility AND realised volatility values of the previous day.

## The Results
The results of the experiments showed GJR-GARCH to be one of the best models for forecasting the volatility in both pairs, however for the EUR/GBP pair, the regression model containing implied and realised volatility values also performed very well.

In addtion to finding out which model provides the best forecasts of volatility, some other intersting details were found. For one, a statistical break down of the coefficients in the fitted GARCH-type models suggested the existence of asymmetrical returns in the EUR/GBP pair but not in the GBP/USD pair. In addtion to this, we also saw that in the in-sample data, modelling the residuals in the GARCH models as a Student's T-distribution provided a better fit that a standard normal distribution.


## Future Work
The fact that the predictive performance of the models varies between the two datasets, suggest that there is not one model that definitively provides the best fit across all assets and that the results of the experiments carried out in this project are only relevant to the GBP/USD and EUR/GBP currency pairs. As such it would be interesting in future to repeat the undertaken analysis on various other currency pairs, including those not containing the pound. One factor that could be interesting is to compare the performance of the models across both major and minor currency pairs to see if the different characteristics between them (such as lower trading volume in minor pairs) has an effect on the models suitability.


## Feedback
The final mark received for the whole project was **82%**. Feedback from my supervisor, [Prof. Robert Sollis](https://www.ncl.ac.uk/business/people/profile/robertsollis.html), can be seen [here](https://github.com/boyla950/predicting-the-pound/blob/main/feedback.pdf).

> By [boyla950](https://github.com/boyla950).
