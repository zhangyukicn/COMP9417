import numpy as np
import pandas as pd

# The data used for training should be strictly in this format
# -root directory
# -data
# --order_products__prior.csv
# --order_products__train.csv
# --orders.csv
# --products.csv
# -code...

prior_address = "./data/order_products__prior.csv"
train_address = "./data/order_products__train.csv"
orders_address = "./data/orders.csv"
products_address = "./data/products.csv"

print("-------Well, the data proccsing costs lots of time so I decide to print something-------")
print("-------Not only to make it not so boring, but also let you know our little program is still working-------")
print("-------This artificial idiot is going to read files from your disk-------")
print("-------But plz give it some time as the files are a little bit large-------")
prior = pd.read_csv(prior_address)
train = pd.read_csv(train_address)
orders = pd.read_csv(orders_address)
products = pd.read_csv(products_address)
print("-------load file completed-------")

# update 1, memory cost is one of the major problems in lightgbm
# so we are going to transfer the default type from int64 to np.uint16 or np.uint32
# based on data_analysis.ipynb we learned each colunm's maximum value and the maximum value 
# of usigned int8, unsigned int16 and unsigned int32
print("-------convert type start-------")

products["product_id"] = products["product_id"].astype(np.uint16)
products["aisle_id"] = products["aisle_id"].astype(np.uint8)
products["department_id"] = products["department_id"].astype(np.uint8)
#axis 默认为0，指删除行，因此删除columns时要指定axis=1；
products = products.drop(['product_name'],axis = 1)
products.info()
# in products we noticed that the "products name" has the same function with product_id
# and id is much more simple to handle, so we are going to remove "products name" column

prior["order_id"] = prior["order_id"].astype(np.uint32)
prior["product_id"] = prior["product_id"].astype(np.uint16)
prior["add_to_cart_order"] = prior["add_to_cart_order"].astype(np.uint8)
prior["reordered"] = prior["reordered"].astype(np.uint8)

train["order_id"] = train["order_id"].astype(np.uint32)
train["product_id"] = train["product_id"].astype(np.uint16)
train["add_to_cart_order"] = train["add_to_cart_order"].astype(np.uint8)
train["reordered"] = train["reordered"].astype(np.uint8)

orders["order_id"] = orders["order_id"].astype(np.uint32)
orders["user_id"] = orders["user_id"].astype(np.uint32)
orders["order_number"] = orders["order_number"].astype(np.uint8)
orders["order_dow"] = orders["order_dow"].astype(np.uint8)
orders["order_hour_of_day"] = orders["order_hour_of_day"].astype(np.uint8)

print("-------convert type finished-------")
print("-------Compute extra feature start-------")

# For each specific product, calculate how many times it has been bought
# if the RAM cost is too high, go adjust the variable type here
products_tmp = pd.DataFrame()
products_tmp['product_ordered_num'] = prior.groupby(prior.product_id).size().astype(np.uint32)
products_tmp['product_reordered_num'] = prior['reordered'].groupby(prior.product_id).sum().astype(np.float32)
products_tmp['product_reordered_rate'] = (products_tmp.product_reordered_num / products_tmp.product_ordered_num).astype(np.float32)

products = products.join(products_tmp, on='product_id')
#products.set_index('product_id', drop=False, inplace=True)
del products_tmp

orders.set_index('order_id', inplace=True, drop=False)
prior = prior.join(orders, on='order_id', rsuffix='tmp')
#prior.info()
prior.drop(["order_idtmp"],axis = 1,inplace = True)

user_tmp = pd.DataFrame()
user_tmp['orders_frequency_days'] = orders.groupby('user_id')['days_since_prior_order'].mean().astype(np.float32)
user_tmp['total_orders'] = orders.groupby('user_id').size().astype(np.int16)
#user_tmp.info()
#print(user_tmp.head(5))
user_tmp['item_sum'] = prior.groupby('user_id').size().astype(np.int16)
user_tmp['products_list'] = prior.groupby('user_id')['product_id'].apply(set)
user_tmp['unique_products_num'] = prior.groupby('user_id')['product_id'].nunique()

user_tmp['item_sum'] = prior.groupby('user_id').size().astype(np.int16)
user_tmp['items_order'] = (user_tmp.item_sum / user_tmp.total_orders).astype(np.float32)

#print(user_tmp.head(5))
#print(products.head(5))
#print(orders.head(5))
#print(prior.info())
print("finished")

products.info()

#user_tmp.info()
#print(user_tmp.head(1))
user_tmp.info()
print(user_tmp.products_list[1])

products["product_ordered_num"] = products["product_ordered_num"].astype(np.float32)
products.info()

orders.info()
print(orders.head(5))

orders_tmp = prior.copy()

print(orders_tmp.head(5))
orders_tmp.info()

orders_tmp1 = pd.DataFrame()
orders_tmp1["userproduct_order_num"] = orders_tmp.groupby(["user_id","product_id"]).size()
orders_tmp1["user_last_order"] = orders_tmp.groupby(["user_id","product_id"])["order_id"].agg('max')
print(orders_tmp1.head(5))

orders_tmp = pd.merge(orders_tmp,orders_tmp1,how = "inner",on = ["user_id","product_id"])
orders_tmp.info()
print(orders_tmp.head(5))

orders_tmp["userproduct_order_num"] = orders_tmp["userproduct_order_num"].astype(np.uint16)
orders_tmp["days_since_prior_order"] = orders_tmp["days_since_prior_order"].astype(np.float32)
orders_tmp.info()

del orders_tmp1

train_set = orders[orders.eval_set =='train']
train_set["days_since_prior_order"] = train_set["days_since_prior_order"].astype(np.float32)
train_set.info()

order_id_list = []
product_id_list = []
label = []
train.set_index(["order_id","product_id"],inplace = True, drop = False)
train.info()

train.info()


for data in train_set.itertuples():
    product_tmp_list = []
    order_id = data.order_id
    user_id = data.user_id
    for i in user_tmp.products_list[user_id]:
        product_id_list.append(i)
        order_id_list.append(order_id)
        product_tmp_list.append(i)
    if(len(order_id_list) != len(product_id_list)):
        print(len(order_id_list))
        print(len(product_id_list))
    label += [(order_id,product) in train.index for product in product_tmp_list]

train_dict = {'order_id': order_id_list,'product_id':product_id_list}
print(len(order_id_list))
print(len(product_id_list))

train_set = pd.DataFrame(train_dict)
train_set["user_id"] = train_set.order_id.map(orders.user_id)
train_set['user_orders_num'] = train_set.user_id.map(user_tmp.total_orders)
train_set['user_history_item_num'] = train_set.user_id.map(user_tmp.item_sum)
train_set['user_history_different_item_num'] = train_set.user_id.map(user_tmp.unique_products_num)
train_set['user_order_frequency_bydays'] = train_set.user_id.map(user_tmp.orders_frequency_days)
train_set['user_average_order'] = train_set.user_id.map(user_tmp.items_order)
train_set["order_hour_of_day"] =  train_set.order_id.map(orders.order_hour_of_day)
train_set["days_since_prior_order"] = train_set.order_id.map(orders.days_since_prior_order)
train_set["day_weight"] = train_set.days_since_prior_order/train_set.user_order_frequency_bydays
train_set["aisle_id"] = train_set.product_id.map(products.aisle_id)
train_set["department_id"] = train_set.product_id.map(products.department_id)
train_set["product_ordered_num"] = train_set.product_id.map(products.product_ordered_num)
train_set["product_reordered_num"] = train_set.product_id.map(products.product_reordered_num)
train_set["product_reordered_rate"] = train_set.product_id.map(products.product_reordered_rate)
train_set.info()

orders_tmp["sub_index"] = orders_tmp.user_id*1000000+orders_tmp.product_id
orders_tmp.info()

train_set["sub_index"] = train_set.user_id*1000000+train_set.product_id
train_set["userproduct_order_num"] = train_set.sub_index.map(orders_tmp.userproduct_order_num)
train_set["userproduct_order_rate"] = train_set.userproduct_order_num/train_set.user_orders_num
train_set["userproduct_last_order"] = train_set.sub_index.map(orders_tmp.user_last_order)
train_set.info()

train_set["userproduct_reorder_rate"] = train_set.product_ordered_num/train_set.user_orders_num
train_set.drop(["sub_index"],axis = 1, inplace = True)
train_set.to_csv("./data/train_set.csv")

label_dict = {"label":label}
label_df = pd.DataFrame(label_dict)
label_df.to_csv("./data/label.csv")

train_set = orders[orders.eval_set =='test']
train_set["days_since_prior_order"] = train_set["days_since_prior_order"].astype(np.float32)
train_set.info()
order_id_list = []
product_id_list = []
train.set_index(["order_id","product_id"],inplace = True, drop = False)
i = 0
for data in train_set.itertuples():
    i+=1
    if(i%10000 == 0):
        print(i)
    product_tmp_list = []
    order_id = data.order_id
    user_id = data.user_id
    for i in user_tmp.products_list[user_id]:
        product_id_list.append(i)
        order_id_list.append(order_id)
        product_tmp_list.append(i)
    if(len(order_id_list) != len(product_id_list)):
        print(len(order_id_list))
        print(len(product_id_list))
train_dict = {'order_id': order_id_list,'product_id':product_id_list}
train_set = pd.DataFrame(train_dict)
train_set["user_id"] = train_set.order_id.map(orders.user_id)
train_set['user_orders_num'] = train_set.user_id.map(user_tmp.total_orders)
train_set['user_history_item_num'] = train_set.user_id.map(user_tmp.item_sum)
train_set['user_history_different_item_num'] = train_set.user_id.map(user_tmp.unique_products_num)
train_set['user_order_frequency_bydays'] = train_set.user_id.map(user_tmp.orders_frequency_days)
train_set['user_average_order'] = train_set.user_id.map(user_tmp.items_order)
train_set["order_hour_of_day"] =  train_set.order_id.map(orders.order_hour_of_day)
train_set["days_since_prior_order"] = train_set.order_id.map(orders.days_since_prior_order)
train_set["day_weight"] = train_set.days_since_prior_order/train_set.user_order_frequency_bydays
train_set["aisle_id"] = train_set.product_id.map(products.aisle_id)
train_set["department_id"] = train_set.product_id.map(products.department_id)
train_set["product_ordered_num"] = train_set.product_id.map(products.product_ordered_num)
train_set["product_reordered_num"] = train_set.product_id.map(products.product_reordered_num)
train_set["product_reordered_rate"] = train_set.product_id.map(products.product_reordered_rate)
orders_tmp["sub_index"] = orders_tmp.user_id*1000000+orders_tmp.product_id
train_set["sub_index"] = train_set.user_id*1000000+train_set.product_id
train_set["userproduct_order_num"] = train_set.sub_index.map(orders_tmp.userproduct_order_num)
train_set["userproduct_order_rate"] = train_set.userproduct_order_num/train_set.user_orders_num
train_set["userproduct_last_order"] = train_set.sub_index.map(orders_tmp.user_last_order)
train_set["userproduct_reorder_rate"] = train_set.product_ordered_num/train_set.user_orders_num
train_set.drop(["sub_index"],axis = 1, inplace = True)
train_set.to_csv("./data/test_set.csv")
