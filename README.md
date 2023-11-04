# Predicting the Pound (Master's Level Dissertation Project)
Predicting the Pound is the project I undertook as part of my dissertation module whilst studying for my MSc Finance degree at Newcastle University Business School. The focus of the project was to investigate the performance of various volatility models, specifically GARCH-type and Implied Volatility, when attempting to forecast the volatility of the GBP/USD and EUR/GBP currency pairs. This repository contains the various Python scripts used to evaluate the models and my final submitted [dissertation paper](https://github.com/boyla950/predicting-the-pound/blob/main/paper.pdf).

## The Goal

## The Dataset
In this project, I downloaded daily price data for each of the currency pairs from Yahoo Finance from a period ranging between June 2018 to June 2023. The raw data contained the open, close, high and close prices for each currency pair as well as the Adjusted Close prices and the daily trading volume. This data was then cleaned to focus solely on the close prices each day.

As volatility is not directly observable, it was necessary to select an appropriate proxy for its value. While there is much discourse in the academic literature as to the most appropriate proxy for this, we opt to use the 20-day standard deviation of returns. This results in much less noisy data than the absolute values of daily returns (another potential proxy), making it much more useful for forecasting.


## The Models


## The Results


## Future Work


## Feedback
The final mark received for the whole project was **82%**. Feedback from my supervisor, [Dr Robert Sollis](https://www.ncl.ac.uk/business/people/profile/robertsollis.html), can be seen [here]().

> By [boyla950](https://github.com/boyla950).
