########################################################################################################################
# ______________________________________________ REAL TIME EARTHQUAKE APP ______________________________________________
########################################################################################################################



# ----------------------------------------------------------------------------------------------------------------------
# CONTENTS
# ----------------------------------------------------------------------------------------------------------------------

# 1. EDA
#   Import Libraries
#   Display Options
#   Warnings Ignore
#   Import Dataset
#   Check Df
#   Grab Col Names
#   Outlier Thresholds
#   Check Outlier
#   Grab Outliers
#   Remove Outlier
#   Replace With Thresholds
# 2. FEATURE ENGINEERING
#   Missing Values Table
#
# 3. MODELLING
#
# 4. FEATURE IMPORTANCE
#
# 5. HYPERPARAMETER OPTIMIZATION
#
# 6. FINAL MODEL
#


print("""
      \n
      _____________________________________________ REAL TIME EARTHQUAKE APP ___________________________________________
        
        --------
        CONTENTS
        --------
        
        1. EDA
            Import Libraries
            Display Options
            Warnings Ignore
            Load Df
            Check Df
            *****
            *****
        2. FEATURE ENGINEERING
            *****
            *****
        3. MODELLING
            *****
            *****
        4. FEATURE IMPORTANCE
            *****
            *****
        5. HYPERPARAMETER OPTIMIZATION
            *****
            *****
        6. FINAL MODEL
            *****
            *****
      __________________________________________________________________________________________________________________
      \n
      """)



##########################################
# STEP 1 : EDA
##########################################
print("\n", "STEP 1 : EDA", "\n", "=" * 30, "\n")


# Import Libraries
# ----------------------------------------
print("\n", "Import Libraries", "\n", "_" * 30, "\n")
import pandas as pd

# Display Options
# ----------------------------------------
print("\n", "Display Options", "\n", "_" * 30, "\n")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


# Warnings Ignore
# ----------------------------------------
print("\n", "Warnings Ignore", "\n", "_" * 30, "\n")
import warnings
warnings.simplefilter(action="ignore")


# Load Df
# ----------------------------------------
print("\n", "Load Df", "\n", "_" * 30, "\n")
def load_df():
    data = pd.read_csv("data/depremler_hava_nasa.csv")
    return data
df = load_df()


# Check Df
# ----------------------------------------
print("\n", "Check Df", "\n", "_" * 30, "\n")
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    # print("##################### Quantiles #####################")
    # print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
check_df(df)


# Required Changes
# ----------------------------------------
print("\n", "Required Changes", "\n", "_" * 30, "\n")
df['time'] = pd.to_datetime(df['time'])
# df.dropna(inplace=True)
# df = df.drop(["id"], axis=1)


# *****
# ----------------------------------------
print("\n", "*****", "\n", "_" * 30, "\n")



# *****
# ----------------------------------------
print("\n", "*****", "\n", "_" * 30, "\n")



# *****
# ----------------------------------------
print("\n", "*****", "\n", "_" * 30, "\n")



# *****
# ----------------------------------------
print("\n", "*****", "\n", "_" * 30, "\n")



# *****
# ----------------------------------------
print("\n", "*****", "\n", "_" * 30, "\n")




##########################################
# STEP 2 : FEATURE ENGINEERING
##########################################
print("\n", "STEP 2 : FEATURE ENGINEERING", "\n", "=" * 30, "\n")


# *****
# ----------------------------------------
print("\n", "*****", "\n", "_" * 30, "\n")




##########################################
# STEP 3 : MODELLING
##########################################
print("\n", "STEP 3 : MODELLING", "\n", "=" * 30, "\n")


# *****
# ----------------------------------------
print("\n", "*****", "\n", "_" * 30, "\n")




##########################################
# STEP 4 : FEATURE IMPORTANCE
##########################################
print("\n", "STEP 4 : FEATURE IMPORTANCE", "\n", "=" * 30, "\n")


# *****
# ----------------------------------------
print("\n", "*****", "\n", "_" * 30, "\n")




##########################################
# STEP 5 : HYPERPARAMETER OPTIMIZATION
##########################################
print("\n", "STEP 5 : HYPERPARAMETER OPTIMIZATION", "\n", "=" * 30, "\n")


# *****
# ----------------------------------------
print("\n", "*****", "\n", "_" * 30, "\n")




##########################################
# STEP 6 : FINAL MODEL
##########################################
print("\n", "STEP 6 : FINAL MODEL", "\n", "=" * 30, "\n")


# *****
# ----------------------------------------
print("\n", "*****", "\n", "_" * 30, "\n")




print("\n", "-" * 50, "PROJECT FINISHED", "-" * 50, "\n")

