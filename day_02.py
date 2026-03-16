import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


days = np.arange(30)
exercises = {"Bench":(60,0.01), "Squat":(100,0.02), "Deadlift":(90,0.03)}
rows =[]
for exercise, (base, prog) in exercises.items():
    noise = np.random.normal(0, 2, len(days))
    weights = base + days * prog + noise
    for weight, day in zip(weights, days):
        rows.append({
                    "Date": pd.Timestamp("2000-01-01") + pd.Timedelta(days = day),
                    "Exercise": exercise,
                    "Weight": round(weight,2),
                    "Sets": np.random.randint(3,5),
                    "Reps": np.random.randint(6,10)\
                            if weight < base + days.mean() * prog \
                            else np.random.randint(4,8)
                    })
df = pd.DataFrame(rows)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Exercise", "Date"])
df["Days"] = (df["Date"] - df["Date"].min()).dt.days

#Volume
df["Volume"] = df["Weight"] * df["Reps"] * df["Sets"]

#OneRM
df["OneRM"] = df["Weight"] * (1 + df["Reps"] / 30)

#Intensity
df["Intensity"] = df["Weight"] / df["OneRM"]
df["Trainingload"] = df["Intensity"] * df["Volume"] * 0.01
df["Acuteload"] = df.groupby("Exercise")["Trainingload"]\
                    .transform(lambda x:x.rolling(7,min_periods=1).mean())
df["Chronicload"] = df.groupby("Exercise")["Trainingload"]\
                    .transform(lambda x:x.rolling(21,min_periods=1).mean())
df["LoadRatio"] = df["Acuteload"] / df["Chronicload"]


#Total Volume 
total_volume_per_date = df.groupby("Date")["Volume"].sum()
total_volume_per_exercise = df.groupby("Exercise")["Volume"].sum()

#Fatigue
df["Fatigue"] = df.groupby("Exercise")["Volume"]\
                / df.groupby("Exercise")["Volume"].transform(lambda x:x.mean())

#Top 3 Exercise
top_volume_per_date = total_volume_per_date.sort_values(ascending=False).head(3)

#Rolling Workload
df["RollingWorkload"] = df.groupby("Exercise")["Volume"]\
                        .transform(lambda x:x.rolling(7, min_periods=1).sum())
df["ChronicWorkload"] = df.groupby("Exercise")["Volume"]\
                        .transform(lambda x:x.rolling(21, min_periods=1).mean())
df["AnotherFatigueIndicator"] = df["RollingWorkload"] / df["ChronicWorkload"]

#Visualization
def plt_feature(feature):
    plt.figure(figsize=(10,5))
    for name, group in df.groupby("Exercise"):
        plt.plot(group["Date"], group[feature], marker='o', label=name)
        maximum = group.loc[group[feature].idxmax()]
        plt.scatter(maximum["Date"], maximum[feature], color="red", s=100, zorder=5)
    plt.title(f"{feature} progress over time")
    plt.xlabel("Date")
    plt.ylabel(feature)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.show()
plt_feature("Weight")
plt_feature("Fatigue")
plt_feature("RollingWorkload")
plt_feature("AnotherFatigueIndicator")
'''
#Prediction
#from sklearn.linear_model import LinearRegression
def pred_feature(feature):
    model = LinearRegression()
    pred = {}
    for name, group in df.groupby("Exercise"):
        x = group[["Days"]]
        y = group[feature]
        model.fit(x,y)
        future_days = pd.DataFrame({
                        "Days":np.arange(group["Days"].max()+1,group["Days"].max()+8)
                                    })
        prediction = model.predict(future_days)
        pred[name] = prediction
        future_dates = group["Date"].max() + pd.to_timedelta(np.arange(1,8), unit="D")
        plt.plot(group["Date"], group[feature], label=f"{name} actual")
        plt.plot(future_dates, prediction, linestyle="--", label=f"{name} forecast")
        for day, value in enumerate(prediction, start=1):
            print(f"{name} day + {day} predicted {feature}: {value:.2f}")
'''

