import pandas as pd
lgb_pred = pd.read_csv('../data/predictions/prediction_lightgbm.csv')
xgb_pred = pd.read_csv('../data/predictions/prediction_xgboost.csv')
com_pred = pd.merge(lgb_pred, xgb_pred, on = "claim_number", how = "left")
com_pred['fraud'] = 0.6 * com_pred['fraud_x'] + 0.4 * com_pred['fraud_y']
com_pred.drop(['fraud_x', 'fraud_y'], axis = 1, inplace = True)
com_pred = com_pred.set_index('claim_number')
com_pred.to_csv('../data/predictions/combined_predictions.csv')