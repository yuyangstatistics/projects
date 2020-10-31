#####  Import packages  #####
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn import model_selection
from skopt import BayesSearchCV
from matplotlib import pyplot
from xgboost import plot_importance
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, VotingClassifier)
from mlxtend.classifier import StackingCVClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb

from random import sample
import random
from statistics import mean


#####  Transform dataset for XGBoost and LightGBM  #####

### XGBoost
#  Import Data 
train = pd.read_csv("../data/train_data_clean_4_grouped.csv")
test = pd.read_csv("../data/test_data_clean_4_grouped.csv")

#  Encode gender and living status and state 
train["living_status"] = pd.Categorical(train["living_status"])
train["gender"] = np.where(train["gender"].str.contains("M"), 1, 0)
train["living_status"] = np.where(train["living_status"].str.contains("Rent"), 1, 0)

test["living_status"] = pd.Categorical(test["living_status"])
test["gender"] = np.where(test["gender"].str.contains("M"), 1, 0)
test["living_status"] = np.where(test["living_status"].str.contains("Rent"), 1, 0)

# one-hot encoding for site of state
state_dummies = pd.get_dummies(test['state'], 
                                  prefix='state', drop_first=True)
test = pd.concat([test, state_dummies], axis=1)
test.drop(["state"], axis=1, inplace=True)

# one-hot encoding for site of state
state_dummies = pd.get_dummies(train['state'], 
                                  prefix='state', drop_first=True)
train = pd.concat([train, state_dummies], axis=1)
train.drop(["state"], axis=1, inplace=True)


# Drop month, day and year data, drop vehicle color, zipcode, claim_date, claim_number and SP_Index  #####
train.drop(["claim_month_january", "claim_month_february", "claim_month_march", "claim_month_may", 
              "claim_month_june", "claim_month_july", "claim_month_august", "claim_month_september", 
              "claim_month_october", "claim_month_november", "claim_month_december", 
              "claim_day_monday", "claim_day_tuesday", "claim_day_wednesday", "claim_day_thursday", 
               "claim_day_saturday", "claim_day_sunday", "claim_year", "claim_day", 
              "zip_code", "claim_date", "claim_number", 'SP_Index', "vehicle_color_blue", 
               "vehicle_color_gray", "vehicle_color_other", "vehicle_color_red", 
              "vehicle_color_silver", "vehicle_color_white"], axis =1, inplace=True)

test.drop(["claim_month_january", "claim_month_february", "claim_month_march", "claim_month_may", 
              "claim_month_june", "claim_month_july", "claim_month_august", "claim_month_september", 
              "claim_month_october", "claim_month_november", "claim_month_december", 
              "claim_day_monday", "claim_day_tuesday", "claim_day_wednesday", "claim_day_thursday", 
               "claim_day_saturday", "claim_day_sunday", "claim_year", "claim_day", 
              "zip_code", "claim_date", "claim_number", 'SP_Index', "vehicle_color_blue", 
               "vehicle_color_gray", "vehicle_color_other", "vehicle_color_red", 
              "vehicle_color_silver", "vehicle_color_white"], axis =1, inplace=True)


#  Add saftyrating/(number of past claim) feature 
train['per_saftyrating'] = train['safty_rating']/(train['past_num_of_claims']+1)
test['per_saftyrating'] = test['safty_rating']/(test['past_num_of_claims']+1)


# Delete some fraud_mean variables  
train.drop(["fraud_gender", "fraud_marital_status", "fraud_high_education_ind", "fraud_address_change_ind", 
              "fraud_living_status", "fraud_zip_code", "fraud_claim_date", "fraud_witness_present_ind", 
              "fraud_policy_report_filed_ind", "fraud_accident_site", "fraud_channel", "fraud_vehicle_category",
           "fraud_vehicle_color", "fraud_state","Unem_rate"],
              axis = 1, inplace = True)
test.drop(["fraud_gender", "fraud_marital_status", "fraud_high_education_ind", "fraud_address_change_ind", 
              "fraud_living_status", "fraud_zip_code", "fraud_claim_date", "fraud_witness_present_ind", 
              "fraud_policy_report_filed_ind", "fraud_accident_site", "fraud_channel", "fraud_vehicle_category",
          "fraud_vehicle_color", "fraud_state", "Unem_rate"],
              axis = 1, inplace = True)
train = train.filter(regex="^(?!state_).*$")
test = test.filter(regex="^(?!state_).*$")

train_xgb = train.copy()
test_xgb = test.copy()


### LightGBM
# read full training data set
df_train = pd.read_csv('../data/train_data_clean_5_grouped.csv')
gender_dummies = pd.get_dummies(df_train['gender'], 
                             prefix = 'gender', drop_first = True)
df_train = pd.concat([df_train, gender_dummies], axis = 1)
df_train.drop(["gender"], axis = 1, inplace = True)

living_status_dummies = pd.get_dummies(df_train['living_status'], 
                             prefix = 'living_status', drop_first = True)
df_train = pd.concat([df_train, living_status_dummies], axis = 1)
df_train.drop(["living_status"], axis = 1, inplace = True)

state_dummies = pd.get_dummies(df_train['state'], 
                               prefix = 'state', drop_first = True)
df_train = pd.concat([df_train, state_dummies], axis = 1)
df_train.drop(["state"], axis = 1, inplace = True)

df_train = df_train.sample(frac=1, random_state=5)
df_train['new_param'] = df_train.apply(lambda col: col['safty_rating']/(col['past_num_of_claims']+1), axis=1)

df_train.set_index('claim_number', inplace=True)
df_train.sort_index(inplace=True)
df_train.drop(['claim_date','fraud_claim_date','fraud_zip_code',
        "fraud_gender", "fraud_marital_status", 'fraud_accident_site', 'fraud_high_education_ind',
         "fraud_address_change_ind", "fraud_living_status", "fraud_witness_present_ind", 
         "fraud_policy_report_filed_ind", "fraud_channel", "fraud_vehicle_category",
         'fraud_vehicle_color', 'fraud_state', 'SP_Index', 'Unem_rate'], axis = 1, inplace = True)
df_train = df_train.filter(regex="^(?!state_).*$")
df_train = df_train.filter(regex="^(?!vehicle_color_).*$")
df_train = df_train.filter(regex="^(?!claim_day_).*$")
df_train = df_train.filter(regex="^(?!claim_month_).*$")

train_lgb = df_train.copy()

# read full testing data set
df_test = pd.read_csv('../data/test_data_clean_5_grouped.csv')
gender_dummies = pd.get_dummies(df_test['gender'], 
                             prefix = 'gender', drop_first = True)
df_test = pd.concat([df_test, gender_dummies], axis = 1)
df_test.drop(["gender"], axis = 1, inplace = True)

living_status_dummies = pd.get_dummies(df_test['living_status'], 
                             prefix = 'living_status', drop_first = True)
df_test = pd.concat([df_test, living_status_dummies], axis = 1)
df_test.drop(["living_status"], axis = 1, inplace = True)

state_dummies = pd.get_dummies(df_test['state'], 
                               prefix = 'state', drop_first = True)
df_test = pd.concat([df_test, state_dummies], axis = 1)
df_test.drop(["state"], axis = 1, inplace = True)

df_test['new_param'] = df_test.apply(lambda col: col['safty_rating']/(col['past_num_of_claims']+1), axis=1)

df_test.set_index('claim_number', inplace=True)
df_test.sort_index(inplace=True)
df_test.drop(['claim_date','fraud_claim_date','fraud_zip_code',
        "fraud_gender", "fraud_marital_status", 'fraud_accident_site', 'fraud_high_education_ind',
         "fraud_address_change_ind", "fraud_living_status", "fraud_witness_present_ind", 
         "fraud_policy_report_filed_ind", "fraud_channel", "fraud_vehicle_category",
         'fraud_vehicle_color', 'fraud_state', 'SP_Index', 'Unem_rate'], axis = 1, inplace = True)
df_test = df_test.filter(regex="^(?!state_).*$")
df_test = df_test.filter(regex="^(?!vehicle_color_).*$")
df_test = df_test.filter(regex="^(?!claim_day_).*$")
df_test = df_test.filter(regex="^(?!claim_month_).*$")

test_lgb = df_test.copy()


#####  Use CV to get the result  #####
# Set the cost for misclassification
cost_dict = {0: 0, 1: 1, -1: 5}

# Set the seed list for splitting dataset
seed_list = [100, 150, 200, 250, 300, 350]

# Set the parameters of XGBoost and LightGBM
clf = xgb.XGBClassifier(max_depth=3,
            learning_rate=0.06,
            n_estimators=180,
            silent=True,
            objective='binary:logistic',
            gamma=0.35,
            min_child_weight=5,
            max_delta_step=0,
            subsample=0.8,
            colsample_bytree=0.785,
            colsample_bylevel=1,
            reg_alpha=0.01,
            reg_lambda=1,
            scale_pos_weight=1,
            seed=1440,
            missing=None)

lgbm_params = {'boosting_type':'gbdt',  'objective':'binary', 'num_boost_round':800,
               'feature_fraction': .321, 'bagging_fraction':0.50, 'min_child_samples':100,  
               'min_child_weigh':35, 'max_depth':3, 'num_leaves':2, 'learing_rate':0.15,
               'reg_alpha':5, 'reg_lambda': 1.1, 'metric':'auc', 'max_bin': 52,
               'colsample_bytree': 0.9, 'subsample': 0.8, 'is_unbalance': 'true'
}

cost_list = []
thre_list = [0.364]  # to try diffrent range, just modify this code
for threshold in thre_list:
    cost = []
    for seed in seed_list:
        # generate row indexes
        random.seed(seed)
        rindex =  np.array(sample(range(len(train_xgb)), round(0.7 * len(train_xgb))))

        # Split train dataset into training and validation parts
        # train_xgb and test_xgb are for XGBoost, train_lgb and test_lgb are for LightGBM

        training_xgb = train_xgb.iloc[rindex, :]
        validation_xgb = train_xgb.drop(train_xgb.index[rindex])

        training_lgb = train_lgb.iloc[rindex, :]
        validation_lgb = train_lgb.drop(train_lgb.index[rindex])

        # XGBoost
        y_training_xgb = training_xgb["fraud"]
        X_training_xgb = training_xgb.drop("fraud", 1)
        y_validation_xgb = validation_xgb["fraud"]
        X_validation_xgb = validation_xgb.drop("fraud", 1)

        clf.fit(X_training_xgb, y_training_xgb)
        y_validation_prob_xgb = clf.predict_proba(X_validation_xgb)[:,1]

        # LightGBM
        y_training_lgb = training_lgb["fraud"]
        X_training_lgb = training_lgb.drop("fraud", 1)
        y_validation_lgb = validation_lgb["fraud"]
        X_validation_lgb = validation_lgb.drop("fraud", 1)


        lgbm = LGBMClassifier(**lgbm_params)
        lgbm.fit(X_training_lgb.values, y_training_lgb.values)
        y_validation_prob_lgb = lgbm.predict_proba(X_validation_lgb.values)[:,1]

        # Combine the result of two models
        validation_prob = 0.4 * y_validation_prob_xgb + 0.6 * y_validation_prob_lgb

        # Calculate the cost
        validation_pred = (validation_prob > threshold)*1  # a trick to transform boolean into int type
        cost.append(sum([cost_dict[i] for i in (validation_pred - y_validation_xgb)]))
        
    cost_list.append(mean(cost))

min_index = cost_list.index(min(cost_list))
print(thre_list[min_index])
print(cost_list[min_index])


#####  Fraud Classification  #####
# Predict on the test dataset
cost_dict = {0: 0, 1: 1, -1: 5}
test_pred = pd.read_csv('../data/predictions/combined_predictions.csv')
test_pred['fraud'] = (test_pred['fraud'] > 0.364)*1
test_pred = test_pred.set_index('claim_number')
test_pred.to_csv('../data/predictions/fraud_classification.csv')

# Estimate the cost on the test dataset
cost_list[min_index] * len(test_xgb) / (0.3 * len(train_xgb))