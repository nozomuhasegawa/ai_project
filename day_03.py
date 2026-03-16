import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

days = np.arange(30)
exercises = {"Bench":(100,0.01), "Squat":(150,0.02), "Deadlift":(160,0.03)}
rows = []
for exercise, (base, prog) in exercises.items():
    noise = np.random.normal(0,2,len(days))
    weights = base + days * prog + noise
    for day, weight in zip(days, weights):
        rows.append({
                    "Date": pd.Timestamp("2000-01-01") + pd.Timedelta(days = day),
                    "Exercise": exercise,
                    "Sets": np.random.randint(3,5),
                    "Weight": round(weight,2),
                    "Reps": np.random.randint(6,10) if weight < base + days.mean() * prog
                                                    else np.random.randint(4,8)
        })          
df = pd.DataFrame(rows)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Exercise", "Date"])
df["Days"] = (df["Date"] - df["Date"].min()).dt.days
df["Volume"] = df["Sets"] * df["Weight"] * df["Reps"]
df["OneRM"] = df["Weight"] * (1 + df["Reps"] / 30)
df["Intensity"] = df["Weight"] / df["OneRM"]
exercise_group = df.groupby("Exercise")

#Training Load
df["TrainingLoad"] = df["Intensity"] * df["Volume"] * 0.01
df["AcuteLoad"] = exercise_group["TrainingLoad"]\
                    .transform(lambda x:x.rolling(7,min_periods=1).mean())
df["ChronicLoad"] = exercise_group["TrainingLoad"]\
                    .transform(lambda x:x.rolling(21,min_periods=1).mean())
df["LoadRatio"] = df["AcuteLoad"] / df["ChronicLoad"]
df["LoadRatioInterpret"] = np.where(df["LoadRatio"] < 0.7, "TooLittle",\
                            np.where(df["LoadRatio"] >1.5, "High Risk", "Sweet Spot"))
df["AcuteLoad_lag1"] = df.groupby("Exercise")["AcuteLoad"].shift(1)
df["ChronicLoad_lag1"] = df.groupby("Exercise")["ChronicLoad"].shift(1)

#visualization
def plt_feature(feature):
    plt.figure(figsize=(10,5))
    for name, group in exercise_group:
        plt.plot(group["Date"], group[feature], marker='o',label=name)
        ratio = group[group["LoadRatio"] > 1.5]
        plt.scatter(ratio["Date"], ratio[feature], color="red", s=100, zorder=5)
    plt.title(f"{feature} progress over time")
    plt.xlabel("Date")
    plt.ylabel(feature)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
plt_feature("Weight")
plt_feature("Volume")
plt_feature("OneRM")
plt_feature("Intensity")

#Training Dashbord
def training_dashboard(exercise_name):
    group = df[df["Exercise"] == exercise_name]

    plt.figure(figsize=(10,5))

    plt.plot(group["Date"], group["Weight"], marker="o", label="Weight")
    plt.plot(group["Date"], group["AcuteLoad"], label="AcuteLoad")
    plt.plot(group["Date"], group["ChronicLoad"], label="ChronicLoad")

    risk = group[group["LoadRatio"] > 1.5]
    plt.scatter(risk["Date"], risk["Weight"], s=100, label="High LoadRatio")

    plt.title(f"{exercise_name} Training Dashboard")
    plt.xlabel("Date")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


#Prediction
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
def pred_feature(feature):
    model_evaluation = {}
    pred = {}
    for name, group in exercise_group:
        model = LinearRegression()
        x = group[["Days", "AcuteLoad_lag1", "ChronicLoad_lag1"]] 
        y = group[feature]
        
        #model evaluation
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        model_evaluation[name] = (r2, mae)
        print(f"{name} R²: {r2:.3f}")
        print(f"{name} MAE: {mae:.2f}")
        
        #prediction
        future_days = pd.DataFrame({
                                    "Days":np.arange(group["Days"].max()+1, group["Days"].max()+8),
                                    "AcuteLoad_lag1":group["AcuteLoad_lag1"].tail(7).mean(),
                                    "ChronicLoad_lag1": group["ChronicLoad_lag1"].tail(21).mean()
                                    })
        prediction = model.predict(future_days)
        pred[name] = prediction
        future_dates = group["Date"].max() + pd.to_timedelta(np.arange(1,8), unit='D')
        plt.figure(figsize=(10,5))
        plt.plot(future_dates, prediction, linestyle='--', label=f"{name} forecast")
        plt.plot(group["Date"], group[feature], label=f'{name} actual')
        plt.scatter(x_test["Days"], y_pred, color="orange", label="test prediction")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
        
        for day, value in enumerate(prediction, start=1):
            print(f'{name} day + {day} predicted {feature}: {value:.2f}')
    return pred, model_evaluation

future_lifting_weight, weight_eval = pred_feature("Weight")
future_one_rm, one_rm_eval = pred_feature("OneRM")
future_load_ratio, load_ratio_eval = pred_feature("LoadRatio")
'''
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
'''
