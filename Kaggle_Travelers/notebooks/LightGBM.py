#####  Import packages  #####
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, VotingClassifier)
from mlxtend.classifier import StackingCVClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score


#####  Data Manipulation  #####
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
#df_train['prct_payout'] = df_train.apply(lambda col: col['claim_est_payout']/(col['annual_income']), axis=1)
#df_train['age_over_safety'] = df_train.apply(lambda col: col['age_of_driver']/(col['safty_rating']+1), axis=1)
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

#df_test = df_test.sample(frac=1, random_state=5)
df_test['new_param'] = df_test.apply(lambda col: col['safty_rating']/(col['past_num_of_claims']+1), axis=1)
#df_test['prct_payout'] = df_test.apply(lambda col: col['claim_est_payout']/(col['annual_income']), axis=1)
#df_test['age_over_safety'] = df_test.apply(lambda col: col['age_of_driver']/(col['safty_rating']+1), axis=1)

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


#####  Final Model  #####
lgbm_params = {'boosting_type':'gbdt',  'objective':'binary', 'num_boost_round':800,
               'feature_fraction': .321, 'bagging_fraction':0.50, 'min_child_samples':100,  
               'min_child_weigh':35, 'max_depth':3, 'num_leaves':2, 'learing_rate':0.15,
               'reg_alpha':5, 'reg_lambda': 1.1, 'metric':'auc', 'max_bin': 52,
               'colsample_bytree': 0.9, 'subsample': 0.8, 'is_unbalance': 'true'
}

y_train = train_lgb["fraud"]
X_train = train_lgb.drop("fraud", 1)

lgbm = LGBMClassifier(**lgbm_params)
lgbm.fit(X_train.values, y_train.values)
y_preds = lgbm.predict_proba(test_lgb.values)[:,1]

test_lgb['fraud'] = y_preds
results = test_lgb.filter(['fraud'], axis=1)
results.to_csv('results_12_6-3.csv', header=True)