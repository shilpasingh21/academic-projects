# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 23:38:20 2019

@author: Yuhan Zeng
"""

import jsonlines
import pandas as pd


# LOAD DATA
business_id_list = []
city_list = []
state_list = []
categories_list = []
with jsonlines.open('business.json') as reader:
    for json_obj in reader:
        business_id_list.append(json_obj["business_id"])
        city_list.append(json_obj["city"])
        state_list.append(json_obj["state"])
        categories_list.append(json_obj["categories"])

df = pd.DataFrame({'business_id': business_id_list, 
                         'city': city_list,
                         'state': state_list,
                         'categories': categories_list})

df = df.query('city=="Phoenix" & state=="AZ" & categories.str.contains("Restaurants")')
df = df[df['categories'].notnull()]
df = df[df.categories.str.contains("Restaurants")]
df = df[df.categories.str.contains("Chinese|Fast Food|Italian|Pizza|Sandwiches")]

business_id_set = set(df['business_id'].unique())

# Get a dataframe of all the tips for the business_id's that are in the set of restaurants in Phoenix
business_id_list = []
tip_list = []
with jsonlines.open('tip.json') as reader:
    for json_obj in reader:
        if json_obj['business_id'] in business_id_set:
            business_id_list.append(json_obj["business_id"])
            tip_list.append(json_obj["text"])
            
tip_df = pd.DataFrame({'business_id': business_id_list, 'tip': tip_list})
tip_df['tip_combined'] = tip_df.groupby(['business_id'])['tip'].transform(lambda x: ' '.join(x))
tip_df = tip_df.drop(['tip'], axis=1).drop_duplicates()
tip_df.to_csv('phoenix_restaurants_tip.csv')

# Get a dataframe of all the reviews for the business_id's that are in the set of restaurants in Phoenix
business_id_list = []
review_list = []
with jsonlines.open('review.json') as reader:
    for json_obj in reader:
        if json_obj['business_id'] in business_id_set:
            business_id_list.append(json_obj["business_id"])
            review_list.append(json_obj["text"])
            
review_df = pd.DataFrame({'business_id': business_id_list, 'review': review_list})
review_df['review_combined'] = review_df.groupby(['business_id'])['review'].transform(lambda x: ' '.join(x))
review_df = review_df.drop(['review'], axis=1).drop_duplicates()


tp_merged_df = pd.merge(df, tip_df, on='business_id',how='right')
#Convert the categories into a list of labels
tp_merged_df['categories'] = tp_merged_df['categories'].apply(lambda x: x.split(','))
tp_merged_df.to_csv('phoenix_restaurants_tip_merged.csv')

review_merged_df = pd.merge(df, review_df, on='business_id',how='right')
# Convert the categories into a list of labels
review_merged_df['categories'] = review_merged_df['categories'].apply(lambda x: x.split(','))

labels_set = set(["Chinese”, “Fast Food”, “Italian”, “Pizza”, “Sandwiches"])

# Iterate through dataframe and give confined labels to each row
for index, row in review_merged_df.iterrows():
    label = ""
    for word in row["categories"].split(", "):
        if word in labels_set:
            if label == "":
                label += word
            else:
                label += "," + word
    review_merged_df.at[index, 'labels'] = label
    
review_merged_df.to_csv("data_multilabel.csv")

