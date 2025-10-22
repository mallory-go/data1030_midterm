#check
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import shap
import warnings


warnings.simplefilter(action='ignore', category=FutureWarning)
df = pd.read_csv('/Users/mallorygo/Desktop/DATA1030/data1030-fall2025/final reports/ufc-master.csv')
pd.set_option("display.max_columns", None, "display.max_rows", None)

print(df.head())


pd.set_option("display.max_columns", None, "display.max_rows", None)

print(df.dtypes) 
# How many rows and columns do we have in df_merge?
print(df.shape[0]) # number of rows
print(df.shape[1]) # number of rows

print(len(df))

comm_drop = [
'Date','Location'
]
df.drop(comm_drop, axis=1, inplace = True)


print(df.dtypes) 
print(df.shape[0]) # number of rows
print(df.shape[1]) # number of rows

print(len(df))

#Separating the features based on their data types
cat_col = [col for col in df.columns if df[col].dtypes == 'object']
num_col = [col for col in df.columns if col not in cat_col]
print(df['Winner'].value_counts(dropna=False))

#take care of missingness in Winner
print("Unique values in Winner column:", df['Winner'].unique())
# Strip spaces and lowercase all values
df['Winner'] = df['Winner'].astype(str).str.strip().str.capitalize()
print("Cleaned Winner values:", df['Winner'].unique())
df = df[df['Winner'].isin(['Red', 'Blue'])]
print("Rows remaining after filtering:", len(df))

unique_categorieswc = df['WeightClass'].unique()
print(unique_categorieswc)

from sklearn.preprocessing import LabelEncoder
print(df['Winner'].value_counts(dropna=False))
# Convert 'Red' and 'Blue' winner string to binary label (1 = Red wins, 0 = Blue wins)
df = df[df['Winner'].isin(['Red', 'Blue'])]  # filter out 'Draw' or 'No Contest'
print("Rows remaining after filtering:", len(df))

cat_col = ['RedFighter', 'BlueFighter', 'WeightClass', 'Gender']

from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()

for col in cat_col:
    if col in df.columns:
        df[col] = enc.fit_transform(df[col].astype(str))
    else:
        print(f"Warning: Column {col} not found in dataframe.")

print(df['Winner'].value_counts())


win_counts = df['Winner'].value_counts()
# Display the frequency table
print("Frequency Table for 'Winner' column:")
print(win_counts)

for column in df.columns:
    if df[column].isnull().sum()!=0:
        print(f"Nan in {column}: {df[column].isnull().sum()}")

#force numeric WeightClass if it's not already
df['WeightClass'] = pd.to_numeric(df['WeightClass'], errors='coerce')

weightclass_labels = {
    0: 'Flyweight',
    1: 'Welterweight',
    2: 'Heavyweight',
    3: 'Featherweight',
    4: 'Light Heavyweight',
    5: 'Catch Weight',
    6: 'Lightweight',
    7: 'Bantamweight',
    8: "Women's Strawweight",
    9: "Women's Flyweight",
    10: "Middleweight",  
    11: "Women's Bantamweight",     
    12: "Women's Featherweight "               
}



df['WeightClassLabel'] = df['WeightClass'].map(weightclass_labels)

#check
missing_wtclass = df['WeightClass'].isnull().sum()
print(f" WeightClass mapped. Missing values: {missing_wtclass}")


# Basic descriptive statistics
print(df.describe())

# Check class balance
sns.countplot(x='Winner', data=df)
plt.title('Fight Winner Distribution (Red = 1, Blue = 0)')
plt.show()

# Missing data heatmap
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Data Heatmap")
plt.show()
print(df["BlueHeightCms"])

import numpy as np
df["BlueHeight"] = df["BlueHeightCms"] / 2.54
df['BlueReach']= df['BlueReachCms'] / 2.54
df['RedHeight'] = df['RedHeightCms'] / 2.54
df['RedReach']= df['RedReachCms'] / 2.54
df.drop(columns = ['BlueHeightCms', 'BlueReachCms', 'RedHeightCms', 'RedReachCms'], axis = 1, inplace = True)

df['ROver35'] = df['RedAge'].apply(lambda x: int(x > 37))
df['BOver35'] = df['BlueAge'].apply(lambda x: int(x > 37))

df['RWinPct'] = df['RedWins'] / (df['RedWins'] + df['RedLosses'])
df['BWinPct'] = df['BlueWins'] / (df['BlueWins'] + df['BlueLosses'])

df['RedStrikingRatio'] = df['RedAvgSigStrLanded'] / (df['RedAvgSigStrLanded'] + df['BlueAvgSigStrLanded'])
df['BlueStrikingRatio'] = df['BlueAvgSigStrLanded'] / (df['RedAvgSigStrLanded'] + df['BlueAvgSigStrLanded'])

df['RedTotalFights'] = df['RedWins'] + df['RedLosses']
df['BlueTotalFights'] = df['BlueWins'] + df['BlueLosses']

df['RedIsGrappler'] = (df['RedAvgTDLanded'] > 3.5).astype(int)
df['BlueIsGrappler'] = (df['BlueAvgTDLanded'] > 3.5).astype(int)

df['RedIsStriker'] = (df['RedAvgSigStrLanded'] > 6.0).astype(int)
df['BlueIsStriker'] = (df['BlueAvgSigStrLanded'] > 6.0).astype(int)

df['RGrapplerVBStriker'] = ((df['RedIsGrappler'] == 1) & (df['BlueIsStriker'] == 1)).astype(int)
df['BGrapplerVRStriker'] = ((df['BlueIsGrappler'] == 1) & (df['RedIsStriker'] == 1)).astype(int)

df['RedStrikingEfficiency'] = df['RedAvgSigStrLanded'] * df['RedAvgSigStrPct']
df['BlueStrikingEfficiency'] = df['BlueAvgSigStrLanded'] * df['BlueAvgSigStrPct']

df['RedTDEfficiency'] = df['RedAvgTDLanded'] * df['RedAvgTDPct']
df['BlueTDEfficiency'] = df['BlueAvgTDLanded'] * df['BlueAvgTDPct']

df['RedEffectiveTD'] = 1 / (df['RedAvgTDLanded'] / df['RedAvgSubAtt'])
df['BlueEffectiveTD'] = 1 / (df['BlueAvgTDLanded'] / df['BlueAvgSubAtt'])
df['EffectiveTDDif'] = df['BlueEffectiveTD'] - df['RedEffectiveTD']

df['RedSize'] = df['RedHeight'] + df['RedReach']
df['BlueSize'] = df['BlueHeight'] + df['BlueReach']

df['Favorite'] = (df['RedOdds'] < df['BlueOdds']).astype(int)

df['Favorite'] = df['Favorite'].map({0 : 'Blue', 1 : 'Red'})

df['FavoriteWins'] = (df['Favorite'] == df['Winner']).astype(int)

df.head()

df['draw_diff'] = (df['BlueDraws']-df['RedDraws'])
df['avg_sig_str_pct_diff'] = (df['BlueAvgSigStrPct']-df['RedAvgSigStrPct'])
df['avg_TD_pct_diff'] = (df['BlueAvgTDPct']-df['RedAvgTDPct'])
df['win_by_Decision_Majority_diff'] = (df['BlueWinsByDecisionMajority']-df['RedWinsByDecisionMajority'])
df['win_by_Decision_Split_diff'] = (df['BlueWinsByDecisionSplit']-df['RedWinsByDecisionSplit'])
df['win_by_Decision_Unanimous_diff'] = (df['BlueWinsByDecisionUnanimous']-df['RedWinsByDecisionUnanimous'])
df['win_by_TKO_Doctor_Stoppage_diff'] = (df['BlueWinsByTKODoctorStoppage']-df['RedWinsByTKODoctorStoppage'])
df['odds_diff'] = (df['BlueOdds']-df['RedOdds'])
df['ev_diff'] = (df['BlueExpectedValue']-df['RedExpectedValue'])


blue_win = df[df['Winner'] == 'Blue']
red_win = df[df['Winner'] == 'Red']

blue_win_percent = len(blue_win) / len(df)
red_win_percent = len(red_win) / len(df)
df.head()

df.BlueStance.unique()
#It has one spelling mistake
df['BlueStance'].loc[df['BlueStance']=='Switch '] = 'Switch'
#R_Stance doesn't have this error, so we're cool

print(df['BlueStance'].value_counts())




red_win_percent_high_TDDif = len(red_win[red_win['AvgTDDif'] < -1]) / len(df[df['AvgTDDif'] < -1])
print("Red Win Percentage Increase (w/ +1 more TDs): " + str(red_win_percent_high_TDDif - red_win_percent))

blue_win_percent_high_TDDif = len(blue_win[blue_win['AvgTDDif'] > 1]) / len(df[df['AvgTDDif'] > 1])
print("Blue Win Percentage Increase (w/ +1 more TDs): " + str(blue_win_percent_high_TDDif - blue_win_percent))

print('\n')

red_win_percent_highTD_v_lowTD = len(red_win[(red_win['RedAvgTDLanded'] > 2.5) & (red_win['BlueAvgTDLanded'] < 1)]) / len(df[(df['RedAvgTDLanded'] > 2.5) & (df['BlueAvgTDLanded'] < 1)])
print("Red Win Percentage Increase (w/ Red TD > 2.5, Blue TD < 1): " + str(red_win_percent_highTD_v_lowTD - red_win_percent))

blue_win_percent_highTD_v_lowTD = len(blue_win[(blue_win['BlueAvgTDLanded'] > 2.5) & (blue_win['RedAvgTDLanded'] < 1)]) / len(df[(df['BlueAvgTDLanded'] > 2.5) & (df['RedAvgTDLanded'] < 1)])
print("Blue Win Percentage Increase (w/ Blue TD > 2.5, Red TD < 1): " + str(blue_win_percent_highTD_v_lowTD - blue_win_percent))

print('\n')

red_old_win_percent = len(df[(df['ROver35'] == 1) & (df['Winner'] == 'Red')]) / len(df[df['ROver35'] == 1])
print("Decrease in Win Percentage When Red Fighter > 35 y.o.: " + str(red_old_win_percent - red_win_percent))
blue_old_win_percent = len(df[(df['BOver35'] == 1) & (df['Winner'] == 'Blue')]) / len(df[df['BOver35'] == 1])
print("Decrease in Win Percentage When Blue Fighter > 35 y.o.: " + str(blue_old_win_percent - blue_win_percent))


df.to_csv('df.csv')

print(df['FavoriteWins'].sum() / len(df))
print("Blue Corner Win Percentage: " + str(blue_win_percent))
print("Red Corner Win Percentage: " + str(red_win_percent))


fig, ax = plt.subplots(1, 2)
ax[0].hist(red_win['RedOdds'], alpha=0.7, color='red')
ax[0].hist(red_win['BlueOdds'], alpha=0.7, color='blue')
ax[0].set_title('Odds for Red Wins')
ax[1].hist(blue_win['RedOdds'], alpha=0.7, color='red')
ax[1].hist(blue_win['BlueOdds'], alpha=0.7, color='blue')
ax[1].set_title('Odds for Blue Wins')
plt.show()

sns.scatterplot(x='WinDif', y='LossDif', data=df, hue='RedAvgSigStrLanded')
plt.show()


#calc missing proportions
missing_proportions = df.isnull().sum() / len(df)
print("Proportion of missing values per column:")
print(missing_proportions)

#calc missing percentages
missing_percentages = df.isnull().sum() * 100 / len(df)
print("\nPercentage of missing values per column:")
print(missing_percentages)

#count columns with any missing data
columns_with_missing_data = df.isna().any()
number_of_columns_with_missing_data = columns_with_missing_data.sum()
number_of_columns_with_missing_data

print(df.shape[0]) # number of rows
print(df.shape[1]) # number of rows
#% of missing values per column 
missing_percentage = df.isnull().sum() * 100 / len(df)
print(missing_percentage)

y = df['Winner']
X = df.drop(columns=['Winner'])

#encode target variable labels as integers
le = LabelEncoder()
y = le.fit_transform(y)

plt.figure(figsize=(8, 5))
sns.histplot(df['BlueReach'], kde=True, color='purple')
plt.title('Distribution of Blue Fighter Reach')
plt.xlabel('Reach (cm)')
plt.ylabel('Number of Fighters')
plt.xlim(50, 90)  # limit x-axis range to be reasonable bc who has a 20in reach? 
plt.show()

df['StanceCombo'] = df['RedStance'].astype(str) + ' vs ' + df['BlueStance'].astype(str)

#remove possible leading and trailing spaces in all 'object' columns
ufc_obj = df.select_dtypes(['object'])
df[ufc_obj.columns] = ufc_obj.apply(lambda x: x.str.strip()) 
#categorical 
print(df['Finish'].value_counts())
#filtering
kos_by_round = df[['Finish', 'FinishRound']].query('Finish == "KO/TKO"') # new filtered dataframe
kos_by_round

KOs = sns.countplot(x = kos_by_round['FinishRound']);
plt.title('Total number of KOs from dataset\nby round', pad = 50, weight = 'bold') # \n for line break. S: https://www.python-graph-gallery.com/190-custom-matplotlib-title
plt.xlabel('Round number', labelpad = 20) #bold 
plt.ylabel('KOs', labelpad = 20)
sns.despine() 
plt.ylim([0,1100])
plt.bar_label(KOs.containers[0], weight = 'bold') 
plt.text(x = 3, y = 800, s = 'Most KOs happen in the first round', fontdict = {'size' : 20, 'weight' : 'bold', 'color': 'orange'}, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'));


# Plot: Count of Winners per Matchup type
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='RedIsStriker', hue='Winner', palette='Set2')
plt.title('Match Outcome by RedIsStriker')
plt.xlabel('RedIsStriker)')
plt.ylabel('Number of Wins')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot: Count of Winners per Matchup type
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='BlueIsStriker', hue='Winner', palette='Set2')
plt.title('Match Outcome by BlueIsStriker')
plt.xlabel('BlueIsStriker)')
plt.ylabel('Number of Wins')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




















