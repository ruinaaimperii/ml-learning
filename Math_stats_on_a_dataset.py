import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


pd.options.display.max_rows = 10
pd.options.display.max_columns = 10
pd.options.display.float_format = "{:.1f}".format


training_df = pd.read_csv(filepath_or_buffer="https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
np.random.seed(123456)
print(training_df.describe())
# training_df[["total_rooms", "median_house_value"]].plot(x="total_rooms", y="median_house_value", kind="scatter")
# training_df[["total_bedrooms", "median_house_value"]].plot(x="total_bedrooms", y="median_house_value", kind="scatter")
# training_df[["median_income", "median_house_value"]].plot(x="median_income", y="median_house_value", kind="scatter")
training_df[["total_rooms"]].plot(kind="box")
plt.show()