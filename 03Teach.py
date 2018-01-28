import pandas as pd
import numpy as np

headers = ["age", "workclass", "fnlwgt", "education", "education-num",
           "marital-status", "occupation", "relationship", "race", "sex",
           "capital-gain", "capital-loss", "hours-per-week", "native-country", "class"]

df = pd.read_csv("adult.data.txt", header=None, names=headers, na_values="?")
df.head()

print(df)

cleanup_nums = {"sex": {"Male": 0, "Female": 1}}

df.replace(cleanup_nums, inplace=True)
df.head()
print(df["sex"])
