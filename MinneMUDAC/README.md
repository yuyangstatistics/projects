## MinneMUDAC 2019

This repository contains the key part of the code I have written for the MinneMUDAC project and the presentation slides of our talks.

`MUDAC Fall2019.pdf` introduces the problem and requirements of this data challenge.

`spider.ipynb` crawls futures price data from https://www.mrci.com/ohlc/index.php. Note that there are some duplicate data. For example, data from [20161230](https://www.mrci.com/ohlc/2016/161230.php) is the same as [20170102](https://www.mrci.com/ohlc/2017/170102.php). Need to remove the duplicated values in preprocessing procedure.

`tweet.ipynb` deals with tweet data. It uses LDA model to cluster tweet topics and thus obtains trade relavant and economy relavant tweets. Trump's tweet data is from http://www.trumptwitterarchive.com/archive. The result is shown in `tweetLDA11.html`.


`ts-modeling.ipynb` tries time series modeling. It is mostly based on [mlcourse.ai](https://github.com/Yorko/mlcourse.ai/blob/master/jupyter_english/topic09_time_series/topic9_part1_time_series_python.ipynb).

`models.ipynb` runs models on the stationarized data. Models include linear model, ridge regression, lasso regression, and XGBoost. Also, SHAP is used to interpret XGBoost model.

`last-week-pred.ipynb` uses the last week as validation and make predictions on the last week.

`next-week.ipynb` makes predictions for the desired upcoming week.
