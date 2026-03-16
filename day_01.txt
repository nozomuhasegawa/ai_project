import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


days = np.arange(60)
exercise_params = {"Bench":(80,0.05), "Squat":(100,0.06), "Deadlift":(100,0.07)}
rows = []
for exercise, (base, prog) in exercise_params.items():
    noise = np.random.normal(0,2,len(days))
    weights = base + days * prog + noise
    for day, weight in zip(days, weights):
        rows.append({
                "Date": pd.Timestamp("2000-01-01") + pd.Timedelta(days=day),
                "Exercise": exercise,
                "Weight": round(weight,2),
                "Reps": np.random.randint(6,10) if weight < base + days.mean() * prog \
                        else np.random.randint(4,8)
                })
df = pd.DataFrame(rows)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Exercise", "Date"])
df["Days"] = (df["Date"] - df["Date"].min()).dt.days
df["Volume"] = df["Weight"] * df["Reps"]

#RollingAvgWeight and RollingVolume
df["RollingAvgWeight"] = df.groupby("Exercise")["Weight"]\
                        .transform(lambda x:x.rolling(7,min_periods=1).mean())
df["RollingAvgVolume"] = df.groupby("Exercise")["Volume"]\
                        .transform(lambda x:x.rolling(7,min_periods=1).mean())
df["RollingChange"] = df.groupby("Exercise")["RollingAvgWeight"].diff().abs()

#Plateau Streak                        
df["PlateauDetect"] = (
    df.groupby("Exercise")["RollingChange"].transform(lambda x: x.rolling(3, min_periods=3).mean())) < 0.5
plateau = df["PlateauDetect"]
plat_groups = (~plateau).groupby(df["Exercise"]).cumsum()
df["PlateauStreak"] = plateau.groupby([df["Exercise"], plat_groups]).cumsum()
df["Plateau"] = df["PlateauStreak"] >= 10

#1RM
df["OneRM"] = df["Weight"] * (1 + df["Reps"] / 30)

#Intensity
df["Intensity"] = df["Weight"] / df["OneRM"]
df["TrainingLoad"] = df["Volume"] * df["Intensity"] * 0.01
df["AcuteLoad"] = df.groupby("Exercise")["TrainingLoad"]\
                   .transform(lambda x: x.rolling(7, min_periods=1).mean())
df["ChronicLoad"] = df.groupby("Exercise")["TrainingLoad"]\
                     .transform(lambda x: x.rolling(28, min_periods=1).mean())
df["LoadRatio"] = df["AcuteLoad"] / df["ChronicLoad"]


#FatigueIndicator
df["WeightChange"] = df.groupby("Exercise")["Weight"].diff()
conditions = [df["WeightChange"] > 0, df["WeightChange"] < 0]
choices = ["Progress", "Regression"]
df["Trend"] = np.select(conditions, choices, default="Plateau")

#Progress Streak
progress = df["Trend"] == "Progress"
prog_groups = (~progress).groupby(df["Exercise"]).cumsum()
df["ProgressStreak"] = progress.groupby([df["Exercise"], prog_groups]).cumsum()

#Fatigue Streak
fatigue = df["Trend"] == "Regression"
fatiguegroups = (~fatigue).groupby(df["Exercise"]).cumsum()
df["FatigueStreak"] = fatigue.groupby([df["Exercise"], fatiguegroups]).cumsum()
df["Fatigue"] = fatigue.groupby("Exercise").transform(lambda x:x.rolling(3,min_periods=1).sum()>=2)

#Fatigue Score
df["FatigueScore"] = (df.groupby("Exercise")["WeightChange"].transform(lambda x: x.rolling(5).mean()))

#Visualization
def plot_feature(feature):
    plt.figure(figsize=(5,8))
    for name, group in df.groupby("Exercise"):
        plt.plot(group["Date"], group[feature], marker='o', label=name)
        streak=group[group["ProgressStreak"] > 0]
        plt.scatter(streak["Date"], streak[feature], s=60, color='blue')
        fatiguestreak=group[group["FatigueStreak"] > 0]
        plt.scatter(fatiguestreak["Date"], fatiguestreak[feature], s=60, color='red')
        plateaustreak=group[group["PlateauStreak"] > 0]
        plt.scatter(plateaustreak["Date"], plateaustreak[feature], s=60, color='yellow', edgecolor='black')
        
    plt.title(f"{feature} Progress Over Time(Progress is blue, fatigue is red, plateau is yellow)")
    plt.xlabel("Date")
    plt.ylabel(feature)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.show()
plot_feature("Weight")
plot_feature("RollingAvgWeight")
plot_feature("RollingAvgVolume")
plot_feature("Volume")
plot_feature("OneRM")
plot_feature("Intensity")
plot_feature("TrainingLoad")



#summary
improved = progress.groupby("Exercise").sum()
fatigue_sum = df.groupby("Exercise")["Fatigue"].sum()
total_volume = df.groupby("Exercise")["Volume"].sum()
most_improve_exercise = improved.idxmax()
most_fatigue_exercise = fatigue_sum.idxmax()
highest_total_volume  = total_volume.idxmax()
print("Most improved exercise:", most_improve_exercise)
print("Most fatigue exercise:", most_fatigue_exercise)
print("Highest training volume:", highest_total_volume)


#Prediction
from sklearn.linear_model import LinearRegression
def predict_feature(feature):
    plt.figure(figsize=(6,4))
    model = LinearRegression()
    prediction = {}
    for name, group in df.groupby("Exercise"):
        x = group[["Days"]]
        y = group[feature]
        model.fit(x,y)
        future_days = pd.DataFrame({"Days":np.arange(group["Days"].max()+1, group["Days"].max()+8)})
        pred = model.predict(future_days)
        prediction[name] = pred
        
        future_dates = group["Date"].max() + pd.to_timedelta(np.arange(1,8), unit="D")
        plt.plot(group["Date"], group[feature], label=f"{name} actual")
        plt.plot(future_dates, pred, linestyle="--", label=f"{name} forcast")

        for day, value in enumerate(pred, start=1):
            print(f"{name} day+{day} predicted {feature}: {value:.2f}")

    plt.title(f"7 Day Prediction for {feature}")
    plt.legend()
    plt.show()
    return prediction

weight_prediction = predict_feature("Weight")
roll_avg_weight_prediction = predict_feature("RollingAvgWeight")
roll_avg_volume_prediction = predict_feature("RollingAvgVolume")
volume_prediction = predict_feature("Volume")
one_rm_prediction = predict_feature("OneRM")
intensity_prediction = predict_feature("Intensity")
training_load_prediction = predict_feature("TrainingLoad")

#Consistency
df["ConsistencyScore"] = df.groupby("Exercise")["Date"].diff().dt.days.fillna(0)
#Average days between exercise
consistency_summary = df.groupby("Exercise")["ConsistencyScore"].mean()
print(consistency_summary)
