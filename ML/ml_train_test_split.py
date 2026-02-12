#train–test split logic
import pandas as pd
from sklearn.model_selection import train_test_split


#splitting dataset

# X = df["final_text"]
# y = df["fraudulent"]

# X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# df  = pd.read_excel(r"C:\Users\aarat\Desktop\Bvoc IT\sem6\proj_s6\main\emscad_cleaned_excel.xlsx")

def split_function(df,text_cols = "text",target_col = "fraudulent",test_size = 0.2, random_state= 42):
    
    X = df[text_cols]
    y = df[target_col]

    X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=test_size,random_state=random_state,stratify=y)

    return X_train , X_test , y_train , y_test 
