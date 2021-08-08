# %%
from pydataset import data
import pandas as pd


# %%
for set in data().dataset_id:
    df = data(set)
    df = df.head()
    strOne = set + '.xlsx'
    df.to_excel(f"C:/Users/marti/OneDrive/Documents/Projects/tensorflow/{strOne}", index = False)

# %%
df = data('BrokenMarriage')
df.head(50)
# %%
data()
# %%
