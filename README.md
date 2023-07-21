# Score between top 5 & 6 in Kaggle

## Background
This Python notebook combines NLP, machine learning and actuarial methods/knowledge to predict the workers compensation insurance claims, see data source: https://www.kaggle.com/competitions/actuarial-loss-estimation/overview
It was a competition held by the Actuaries Institute of Australia, Institute and Faculty of Actuaries and the Singapore Actuarial Society.
My motivation is to use my actuarial and machine learning skills to see how my results compare with winners, though the competition was concluded 2 years ago. The best private score was between top 5 and 6 - you may submit my predictions to Kaggle for verification (yes, your 'late' submission will be scored, but not ranked or disclosed publicly).

## Models
I thought gamma distribution might be relevant for choosing model objective, as it is widely used by actuaries to model claims. Data exploration also showed that claims distribution looked like exponential/gama family. Yet, it turned out that it was not a good fit for machine learning objective. Results speak louder than assumptions. Based on my previous experience, for tabular data, neural networks are not as good as tree-based ensemble methods. So I focused on lightgbm and xgboost, with Bayesian search of hyperparamter tuning by 2 tools: sklearn/skopt and Optuna. Yet, one data field is unstructured - the claims description. I used NLP model by Fasttext to deal with this part (see below). 

## Creative Feature Engineering
Workers compensation ultimate claim payments are often a certain multiple of weekly wages to compensate for loss of income. Hence, the log ratio of ultimate claims cost to weekly wages may reflect the injury severity factor. This can be a better label for claim description to learn to generate new feature, as claims description generally would not consider individual wage level. This also serves as a proxy to adjust for claims inflation. The injury severity factor was categorised by K-Means clustering, then learnt by Fasttext NLP model, which was used to create injury levels for feature engineering.
Another feature to account for bias was created. Different types of claims may induce different levels of bias. As the bias can be extremely large due to its exponential distribution (see Actuarial_explore.ipynb), we compute the difference of log instead, which is equivalent to log of ratio.  Bias was categorised by K-Means clustering, then learnt by Fasttext NLP model, which was used to create bias levels for feature engineering.
Probabilities of injury and bias level classes were used as features.

Outcome: one of these features became top 1 or 2 among different models by 'feature importance'.

## Other Features Added
time_lag: the time difference between the accident and the report of the claim

claim_projected: claims projected by a constant factor to initial claims (accident years were tried but not performing)

Other date/time elements of accidents such as hour, month of the accident date etc. Re-formatting for date/time.

## Data Pre-processing / Cleansing
Missing data in categories were treated as a separated category for machine learning models to learn by itself (also because quite different distribution was observed). This approach is not suitable for all situations, but should work for this context, especially with test data given (though without labels) and similar distribution was observed. The largest claim in training data was identified as an outlier (far larger than other top 10) and removed. Some abnormally small values of wages and initial claims (possibly wrong, e.g. $1 for weekly wages) were replaced by threshold in feature engineering but retained in its own data field.

## Potential Improvements
Other NLP models may be used to tune better, but it might take a few days more from my experience of a few different NLP models, which can be computation intensive. More feature engineering can be explored, such as using the level of prediction (e.g. small or large predicted claim) to differentiate bias levels. Meta learning may be applied to combine different models. As gradient boosting tree methods learn by residuals iteratively, a small portion of predictions may become negative. It was manually corrected (though just small performance change and it didn't affect the 'rank') and could be done more systematically. And tune more for the machine learning models. As I am not really in the competition, I would not spend these efforts further as the score looked pretty already.

## Note
To avoid re-training, download and unzip the NLP models in 'Releases'.


