import pandas as pd
import numpy as np
import lightgbm

label_address = "./data/label.csv"
test_address = "./data/test_set.csv"
train_address = "./data/train_set.csv"
orders_address = "./data/orders.csv"

train = pd.read_csv(train_address)
label = pd.read_csv(label_address)
test = pd.read_csv(test_address)
orders = pd.read_csv(orders_address)

test2 = orders[orders.eval_set =='test']
label["label"] = label["label"].astype(np.float32)
label = label["label"].values.tolist()

param = {"user_orders_num","user_history_item_num","user_history_different_item_num",
        "user_order_frequency_bydays","user_average_order","order_hour_of_day","days_since_prior_order",
         "day_weight","aisle_id","department_id","product_ordered_num","product_reordered_num",
        "product_reordered_rate","userproduct_order_num","userproduct_order_rate","userproduct_reorder_rate"}

train_model = lightgbm.Dataset(train[param],label = label,categorical_feature=['aisle_id', 'department_id'])
param2 = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 96,
    'max_depth': 10,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5
}
print(0)
model = lightgbm.train(param2,train_model,100)
print(1)
result = model.predict(test[param])
print(2)
TRESHOLD = 0.22 
test["result"] = result

d = dict()
for row in test.itertuples():
    if row.result > TRESHOLD:
        try:
            d[row.order_id] += ' ' + str(row.product_id)
        except:
            d[row.order_id] = str(row.product_id)

for order in test2.order_id:
    if order not in d:
        d[order] = 'None'

submission = pd.DataFrame.from_dict(d, orient='index')

submission.reset_index(inplace=True)
submission.columns = ['order_id', 'products']
submission.to_csv('submission.csv', index=False)
