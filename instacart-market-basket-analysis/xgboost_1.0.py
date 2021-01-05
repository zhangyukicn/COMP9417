import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold,train_test_split
import gc


label_address = "./data/label.csv"
test_address = "./data/test_set.csv"
train_address = "./data/train_set.csv"
orders_address = "./data/orders.csv"


train = pd.read_csv(train_address, index_col=0)
label = pd.read_csv(label_address, index_col=0)
test = pd.read_csv(test_address, index_col=0)
orders = pd.read_csv(orders_address)

orders.head()
test2 = orders[orders.eval_set =='test']
label["label"] = label["label"].astype(np.float32)
label = label["label"].values.tolist()

x_train = train
x_test = label
y_train = test


model = XGBClassifier()
gc.collect()
model.fit(x_train, x_test)
y_pred = model.predict(y_train)

predictions = [value for value in y_pred]

test["result"] = predictions

test2 = orders[orders.eval_set =='test']

TRESHOLD = 0.22
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
submission.to_csv('submission_original_xgboost.csv', index=False)

ligbm = pd.read_csv("./submission_original_xgboost.csv", index_col=0)
