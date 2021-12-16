import env
import pandas as pd
import os
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

#Scaling Processers
import sklearn.preprocessing

from datetime import date

#Gets connection to Code Up database using env file
def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------


# Get zillow.csv Data
def get_zillow_data():
    filename = "zillow.csv"

    if os.path.isfile(filename):
        zillow = pd.read_csv(filename)
    else:
        zillow = pd.read_sql("""
#SELECT *
#FROM properties_2017
#INNER JOIN (SELECT parcelid, max(transactiondate) AS max_date, logerror
#FROM predictions_2017
#GROUP BY parcelid, logerror) AS sub1 using(parcelid)
#LEFT JOIN propertylandusetype using(propertylandusetypeid)
#LEFT JOIN airconditioningtype using(airconditioningtypeid)
#LEFT JOIN architecturalstyletype using(architecturalstyletypeid)
#LEFT JOIN buildingclasstype using(buildingclasstypeid)
#LEFT JOIN heatingorsystemtype using(heatingorsystemtypeid)
#LEFT JOIN typeconstructiontype USING (typeconstructiontypeid)
#LEFT JOIN storytype using(storytypeid)
#WHERE properties_2017.latitude IS NOT NULL AND properties_2017.longitude IS NOT NULL AND max_date LIKE '2017%%'
#AND propertylandusetypeid = 261 OR 279;
# Ravinders Code For code up

SELECT prop.*, 
       pred.logerror, 
       pred.transactiondate, 
       air.airconditioningdesc, 
       arch.architecturalstyledesc, 
       build.buildingclassdesc, 
       heat.heatingorsystemdesc, 
       landuse.propertylandusedesc, 
       story.storydesc, 
       construct.typeconstructiondesc 
FROM   properties_2017 prop  
       INNER JOIN (SELECT parcelid,
       					  logerror,
                          Max(transactiondate) transactiondate 
                   FROM   predictions_2017 
                   GROUP  BY parcelid, logerror) pred
               USING (parcelid) 
       LEFT JOIN airconditioningtype air USING (airconditioningtypeid) 
       LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid) 
       LEFT JOIN buildingclasstype build USING (buildingclasstypeid) 
       LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid) 
       LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid) 
       LEFT JOIN storytype story USING (storytypeid) 
       LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid) 
WHERE  prop.latitude IS NOT NULL 
       AND prop.longitude IS NOT NULL AND transactiondate <= '2017-12-31';
""", 
        get_connection('zillow'))
        zillow.to_csv(index = False)
    return zillow


#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
# Remove outliers
def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''

    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------


def prepare_zillow(df):
    ''' Prepare zillow data for exploration'''

    # Fill nullls with zero value
    df[['garagecarcnt', 'garagetotalsqft', 'poolcnt']] = df[['garagecarcnt', 'garagetotalsqft', 'poolcnt']].fillna(0)

    # Drop rest of the null values
    df = df.dropna()

    # Use yearbuilt_years function to transform years into age
    df = yearbuilt_years(df)

    # Rename columns for readability
    df = df.rename(columns = {'fips':'county', 'calculatedfinishedsquarefeet' : 'area', 'bathroomcnt' : 'bathrooms',
                         'bedroomcnt' : 'bedrooms', 'poolcnt' : 'pools', 'garagecarcnt' : 'garages',
                          'taxvaluedollarcnt': 'tax_value'})

    # Create Counties By their codes
    df['LA_County']= df['county'] == 6037
    df['Orange_County']= df['county'] == 6059
    df['Ventura_County']= df['county'] == 6111

    # Set to object type
    df['county'] = df['county'].astype(object)

    # Rename County To showcase counties instead of their numeric id's
    df['county'] = df.county.replace(6059, 'Orange')
    df['county'] = df.county.replace(6037, 'LA')
    df['county'] = df.county.replace(6111, 'Ventura')

    # Replace Counties TRUE/ FALSE values with 0/1's
    df['LA_County'] = df['LA_County'].replace(False, 0)
    df['LA_County'] = df['LA_County'].replace(True, 1)

    df['Orange_County'] = df['Orange_County'].replace(False, 0)
    df['Orange_County'] = df['Orange_County'].replace(True, 1)

    df['Ventura_County'] = df['Ventura_County'].replace(False, 0)
    df['Ventura_County'] = df['Ventura_County'].replace(True, 1)

    # removing outliers
    col_list = ['bedrooms', 'bathrooms', 'area', 'tax_value', 'garages', 'roomcnt', 'garagetotalsqft']
    k = 1.5
    
    df = remove_outliers(df, k, col_list)

    train, validate, test = split_data(df)


    return train, validate, test 

#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------


def split_data(df):
    '''
    Takes in a df
    Returns train, validate, and test DataFrames
    '''
    # Create train_validate and test datasets
    train_validate, test = train_test_split(df, 
                                        test_size=.2, 
                                        random_state=123)
    # Create train and validate datsets
    train, validate = train_test_split(train_validate, 
                                   test_size=.3, 
                                   random_state=123)
    return train, validate, test


#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------

def zillow_scaler(train, validate, test): #X_train , X_validate, X_test

    # 1. create the object
    scaler_min_max = sklearn.preprocessing.MinMaxScaler()

    # 2. fit the object (learn the min and max value)
    # train[['bedrooms', 'taxamount']]
    scaler_min_max.fit(train)

    # 3. use the object (use the min, max to do the transformation)
    # train[['bedrooms', 'taxamount']]
    scaled_bill = scaler_min_max.transform(train)

    train[['bedrooms_scaled', 'taxamount_scaled']] = scaled_bill
    # Create them on the test and validate
    # test[['bedrooms', 'taxamount']]
    test[['bedrooms_scaled', 'taxamount_scaled']] = scaler_min_max.transform(test)
    # validate[['bedrooms', 'taxamount']]
    validate[['bedrooms_scaled', 'taxamount_scaled']] = scaler_min_max.transform(validate)

    return train, validate, test #X_train, X_validate, X_test


#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------

def yearbuilt_years(df):
    df.yearbuilt =  df.yearbuilt.astype(int)
    year = date.today().year
    df['age'] = year - df.yearbuilt
    # dropping the 'yearbuilt' column now that i have the age
    df = df.drop(columns=['yearbuilt'])
    
    return df

#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
# Function to read and wrangle data:
def handle_missing_values(df, prop_required_row = .5, prop_required_col = .7):
    ''' function which takes in a dataframe, required notnull proportions of non-null rows and columns.
    drop the columns and rows columns based on theshold:'''
    
    #drop columns with nulls
    threshold = int(prop_required_col * len(df.index)) # Require that many non-NA values.
    df.dropna(axis = 1, thresh = threshold, inplace = True)
    
    #drop rows with nulls
    threshold = int(prop_required_row * len(df.columns)) # Require that many non-NA values.
    df.dropna(axis = 0, thresh = threshold, inplace = True)
    return df


def remove_columns(df, cols_to_remove):  
    df = df.drop(columns=cols_to_remove)
    return df

def wrangle_zillow_codeup(df):
    #df = pd.read_csv('zillow.csv')
    
    # Already Specificy These two lines with setting typeid to 271 in SQL statement
    # Restrict df to only properties that meet single unit use criteria
    #single_use = [261, 262, 263, 264, 266, 268, 273, 276, 279]
    #df = df[df.propertylandusetypeid.isin(single_use)]
    
    # Restrict df to only those properties with at least 1 bath & bed and 350 sqft area
    df = df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0) & ((df.unitcnt<=1)|df.unitcnt.isnull())\
            & (df.calculatedfinishedsquarefeet>350)]

    # Handle missing values i.e. drop columns and rows based on a threshold
    df = handle_missing_values(df)
    
    # Add column for counties
    df['county'] = np.where(df.fips == 6037, 'Los_Angeles',
                           np.where(df.fips == 6059, 'Orange', 
                                   'Ventura'))    
    # drop columns not needed
    list_of_dropped = ['id','calculatedbathnbr','finishedsquarefeet12','fullbathcnt','propertycountylandusecode','propertylandusetypeid','censustractandblock','propertylandusedesc']
    # 'unitcnt' 'heatingorsystemdesc', 'propertyzoningdesc', 'heatingorsystemtypeid' 'buildingqualitytypeid'

    df = remove_columns(df, list_of_dropped)


    # replace nulls in unitcnt with 1
    # df.unitcnt.fillna(1, inplace = True)
    
    # assume that since this is Southern CA, null means 'None' for heating system
    # df.heatingorsystemdesc.fillna('None', inplace = True)
    
    # replace nulls with median values for select columns
    df.lotsizesquarefeet.fillna(7313, inplace = True)
    # df.buildingqualitytypeid.fillna(6.0, inplace = True)

    # Columns to look for outliers
    df = df[df.taxvaluedollarcnt < 5_000_000]
    df[df.calculatedfinishedsquarefeet < 8000]
    
    # Just to be sure we caught all nulls, drop them here
    df = df.dropna()
    
    return df
#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------

#Do both recieve and clean of zillow data
def wrangle_zillow():

    train, validate, test = prepare_zillow(get_zillow_data())

    return train, validate, test