import pandas as pd
import os
import numpy as np
import env


def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


# Get zillow.csv Data
def get_mall_data():
    filename = "mall_customers.csv"

    if os.path.isfile(filename):
        mall = pd.read_csv(filename)
    else:
        mall = pd.read_sql("""
SELECT * FROM customers
""", 
        get_connection('mall_customers'))
        mall.to_csv(index = False)
    return mall


