##--فراخوانی کتابخانه ها

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import messagebox
from sklearn.ensemble import RandomForestRegressor
##--بارگذاری و پردازش داده ها

columns_am = ['mpg', 'cylinders', 'displacement', 'horsepower', 'wight', 'acceleration', 'model_year', 'origin',
              'car_name']

am = pd.read_csv('../Database_Regressions/auto-mpg.csv', header=None, names=columns_am, sep='\s+')
am.columns = columns_am
# print(am.to_string())
# print(am.shape)
# print(am.isna().sum())
# print(am.info())

##-->.....تبدیل داده های گم شده به NANو حذف آنها
am['horsepower'] = pd.to_numeric(am['horsepower'], errors='coerce')
am.dropna(inplace=True)
##--یافتن out lier data
# #for col in am.select_dtypes(include='number').columns:
##     Q1 = am[col].quantile(0.25)
# #    Q3 = am[col].quantile(0.75)
# #    IQR = Q3 - Q1
# # #
##     lower_bound = Q1 - 1.5 * IQR
##     upper_bound = Q3 + 1.5 * IQR
# # #
##     outliers = am[(am[col] < lower_bound) | (am[col] > upper_bound)]
# # #
##     print(f"ستون: {col}")
##     print(f"  Lower = {lower_bound}")
# #    print(f"  Upper = {upper_bound}")
# #    print(f"  تعداد آوت‌لایرها = {len(outliers)}")
#
##     if len(outliers) == 0:
##         print("  هیچ آوت‌لایری ندارد")
##
##     print("-" * 40)

##--حذف Outlierdata
def remove_outlier(df,column):
    Q1=df[column].quantile(0.25)
    Q3=df[column].quantile(0.75)
    IQR=Q3-Q1

    lower_bound=Q1-1.5*IQR
    upper_bound=Q3+1.5*IQR
    return (df[(df[column]>=lower_bound)&(df[column]<=upper_bound)])
for column in ['mpg','horsepower','acceleration']:
    am=remove_outlier(am,column)


# print(am.to_string())

##__تبدیل ستون origin به دمی
am= pd.get_dummies(am, columns=['origin'], drop_first=True)

# print(am.to_string())
##--جدا کردن ویژگی ها و هدف
x = am.drop(['mpg', 'car_name'], axis=1)
y = am['mpg']

# --تقسیم داده ها به مجموعه های آموزش(X_train)و تست(X_test)#
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
# print(f'sahpe of x_train:{x_train.shape}')
# print(f'x_shape of x_test:{x_test.shape}')
##--استاندارد سازی داده ها
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

##--ساخت مدل رگرسیون خطی
rf_model=RandomForestRegressor(n_estimators=200,random_state=42)
rf_model.fit(x_train_scaled,y_train)

##--پیش بینی مصرف سوخت
y_pred=rf_model.predict(x_test_scaled)
##--ارزیابی مدل
mse=mean_squared_error(y_test,y_pred)

# print(f'Mean Squared Error:{mse:2f}')

joblib.dump(rf_model,'random_forest_model.auto-mpg.joblib')
joblib.dump(scaler,'scaler.auto-mpg.joblib')

# ---------------------------------------

# ==========================
def predict_mpg():
    try:
        # گرفتن مقادیر ورودی
        cylinders = int(cylinders_entry.get())
        displacement = float(displacement_entry.get())
        horsepower = float(horsepower_entry.get())
        weight = float(weight_entry.get())
        acceleration = float(acceleration_entry.get())
        model_year = int(model_year_entry.get())
        origin_val = int(origin_var.get())

        # آماده‌سازی داده برای مدل
        origin_2 = 1 if origin_val == 2 else 0
        origin_3 = 1 if origin_val == 3 else 0

        input_data = np.array([[cylinders, displacement, horsepower, weight, acceleration, model_year, origin_2, origin_3]])
        input_data = scaler.transform(input_data)

        prediction = rf_model.predict(input_data)
        result_label.config(text=f"پیش‌بینی مصرف سوخت (MPG): {prediction[0]:.2f}")
    except ValueError:
        messagebox.showerror("Error", "Please enter all fields correctly!")

# ایجاد پنجره Tkinter
root = tk.Tk()
root.geometry('200x400')
root.title("پیش‌بینی مصرف سوخت خودرو (Auto MPG)")

labels = ["تعداد سیلندرها", "حجم موتور (Displacement)", "توان موتور (Horsepower)",
          "وزن خودرو (Weight)", "شتاب (Acceleration)", "سال مدل (Model Year)"]

entries = []

for label in labels:
    tk.Label(root, text=label + ":").pack()
    entry = tk.Entry(root)
    entry.pack()
    entries.append(entry)

cylinders_entry, displacement_entry, horsepower_entry, weight_entry, acceleration_entry, model_year_entry = entries

tk.Label(root, text="کشور مبدا:").pack()
origin_var = tk.StringVar(value="1")
tk.OptionMenu(root, origin_var, "1","2","3").pack()

tk.Button(root, text="پیش‌بینی مصرف سوخت (MPG)", command=predict_mpg).pack(pady=10)
result_label = tk.Label(root, text="پیش‌بینی مصرف سوخت (MPG): ")
result_label.pack(pady=10)

root.mainloop()



