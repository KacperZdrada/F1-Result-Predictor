import pandas as pd
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

def dataProcessing():
    # Read both results files and transform into pandas dataframe
    qualificationResults = pd.read_csv("../scraper/qualiResults.csv", encoding="cp1252")
    raceResults = pd.read_csv("../scraper/raceResults.csv", encoding="cp1252")
    startingGrid = pd.read_csv("../scraper/startingGrid.csv", encoding="cp1252")
    # Merge the two dataframes into one (matching where every driver qualified and finished
    # Drop the unnecessary column of car numbers
    mergedResults = (pd
                     .merge(qualificationResults, raceResults, on=["Race", "Year", "Driver", "Car Number", "Team"], suffixes=(" Quali", " Race")))

    mergedResults = pd.merge(mergedResults, startingGrid, on=["Race", "Year", "Driver", "Car Number", "Team"]).drop(["Car Number", "QTime"],axis=1)

    # Replace every NC (Not Classfied), DQ (disqualified), RT (Retired), EX (Excluded) position with 26 (lower than any
    # possible position
    mergedResults["Position Race"] = (mergedResults["Position Race"]
                                      .replace("NC", 26)
                                      .replace("DQ", 26)
                                      .replace("RT", 26)
                                      .replace("EX", 26)
                                      .astype(int))
    mergedResults["Position Quali"] = (mergedResults["Position Quali"]
                                       .replace("NC", 26)
                                       .replace("DQ", 26)
                                       .replace("RT", 26)
                                       .replace("EX", 26)
                                       .astype(int))

    # Add binary column for win or no win
    mergedResults["Win"] = mergedResults["Position Race"].apply(lambda x: 1 if x == 1 else 0)

    # Add binary column for race DNF and cumulative team DNFs for a season (up to previous race)
    mergedResults["Race DNF"] = mergedResults["Time"].apply(lambda x: 1 if (x == "DNF" or x == "DNS") else 0)

    # Add column for race number
    mergedResults["Race Number"] = mergedResults.groupby(["Year", "Race"], sort=False).ngroup() + 1

    # Create a new dataframe to store the total amount of DNFs for each team per race
    teamDNFs = mergedResults.groupby(["Year", "Race Number", "Team"]).agg({"Race DNF": "sum"})

    # Add a new column that tracks the cumulative DNFs for each team through the season (excluding current race)
    teamDNFs["Team Season DNF"] = teamDNFs.groupby(["Year", "Team"])["Race DNF"].cumsum() - teamDNFs["Race DNF"]
    teamDNFs = teamDNFs.drop("Race DNF", axis=1)
    # Merge this data with existing dataframe
    mergedResults = mergedResults.merge(teamDNFs, on=["Race Number", "Team"])

    # Add column for cumulative number of driver points excluding the current race
    mergedResults["Season Points"] = (mergedResults.groupby(["Year", "Driver"])["Points"]
                                      .cumsum() - mergedResults["Points"])

    # Create a new dataframe to store the average position for each driver for past 5 races
    last5 = mergedResults[["Year", "Race Number", "Driver", "Position Race", "Season Points", "Team Season DNF"]].copy()
    last5 = last5.sort_values(by=["Driver", "Year", "Race Number"])

    # Create new column to hold the average race position
    last5["Last 5 Race"] = (last5.groupby("Driver")["Position Race"]
                            # Create rolling window of size 6 excluding current row (so window of size 5)
                            .rolling(window=6, min_periods=1, closed="left")
                            .mean()
                            # Fill the first entry with 0 (as no mean here because row is excluded)
                            .fillna(0)
                            .reset_index(level=0, drop=True))

    # Create new column to hold the average race points
    last5["Last 5 Points"] = (last5.groupby(["Driver", "Year"])["Season Points"]
                              # Create rolling window of size 6 excluding current row (so window of size 5)
                              .rolling(window=5, min_periods=1)
                              .apply(lambda x: x.iloc[-1] - x.iloc[0], raw=False)
                              # Fill the first entry with 0 (as no mean here because row is excluded)
                              .fillna(0)
                              .reset_index(level=[0, 1], drop=True))

    # Create new column to hold the average team DNFs
    last5["Last 5 DNF"] = (last5.groupby(["Driver", "Year"])["Team Season DNF"]
                              # Create rolling window of size 6 excluding current row (so window of size 5)
                              .rolling(window=5, min_periods=1)
                              .apply(lambda x: x.iloc[-1] - x.iloc[0], raw=False)
                              # Fill the first entry with 0 (as no mean here because row is excluded)
                              .fillna(0)
                              .reset_index(level=[0, 1], drop=True))

    # Drop unnecessary columns
    print(last5.head(50))
    last5 = last5.drop(["Position Race", "Year", "Season Points", "Team Season DNF"], axis=1)

    mergedResults = mergedResults.merge(last5, on=["Driver", "Race Number"])

    # Fill Null Q2 results with Q1 times
    mergedResults["Q2"] = mergedResults["Q2"].fillna(mergedResults["Q1"])
    # Fill Null Q3 results with Q2 times
    mergedResults["Q3"] = mergedResults["Q3"].fillna(mergedResults["Q2"])

    # Loop over Q1, Q2, and Q3
    for i in range(1,4):

        # Replace DNF (Did Not Finish) and DNS (Did Not Start) times and any other null values
        mergedResults["Q"+str(i)] = (mergedResults["Q"+str(i)]
                               .replace("DNF", "99999:99.999")
                               .replace("DNS", "99999:99.999")
                               .replace("DEL", "99999:99.999")
                               .fillna("99999:99.999")
                               )

        # Convert all string times into float datatype
        mergedResults["Q"+str(i)] = mergedResults["Q"+str(i)].apply(lambda x: stringTimetoInt(x))

    # Find the single fastest individual time across all three qualifying sessions for each driver
    mergedResults["Fastest Individual Time"] = mergedResults[["Q1", "Q2", "Q3"]].min(axis=1)

    # Find the single fastest individual time across all three qualifying sessions and all drivers
    mergedResults["Fastest Quali Time"] = (mergedResults.groupby(["Year", "Race Number"])["Fastest Individual Time"]
                                           .transform("min"))

    # Create a new column that has the delta between each driver's fastest qualifying time and the overall
    # qualification fastest time
    mergedResults["Quali Delta"] = mergedResults["Fastest Individual Time"] - mergedResults["Fastest Quali Time"]

    # Create a new column that holds each driver's position in the WDC standings
    mergedResults["WDC"] = mergedResults.groupby("Year").apply(
        lambda season: season.groupby("Race")["Season Points"].rank(ascending=False, method="first")
    ).reset_index(drop=True)

    # Create a new dataframe to store the total amount of points for each team per race
    teamPoints = mergedResults.groupby(["Year", "Race Number", "Team"]).agg({"Season Points": "sum"})

    # Create a new column that holds each team's position in the WCC standings
    teamPoints["WCC"] = teamPoints.groupby("Race Number")["Season Points"].rank(ascending=False, method="first")
    print(teamPoints.head(50))

    # Merge this data with existing dataframe
    mergedResults = mergedResults.merge(teamPoints, on=["Race Number", "Team"], suffixes=("", " Team"))

    # Drop unnecessary columns
    mergedResults = mergedResults.drop(["Q1", "Q2", "Time", "Laps Quali", "Laps Race"], axis=1)

    # Change team names to standard (ignoring engines, sponsors, etc. in names)
    mergedResults["Team"] = mergedResults["Team"].apply(lambda team: updateTeamNames(team))

    # Normalise data
    scaler = StandardScaler()
    mergedResults[["Quali Delta", "Last 5 Points"]] = scaler.fit_transform(mergedResults[["Quali Delta", "Last 5 Points"]])
    print(mergedResults.head(50))
    # One-hot encode categorical data like driver names, team names, and race location
    mergedResults = pd.get_dummies(mergedResults, columns=["Driver", "Team", "Race"], drop_first=True)

    print(qualificationResults.head())
    print(raceResults.head(25))
    print(teamDNFs.head(50))
    print(last5.head(50))
    print(mergedResults.head(50))
    return mergedResults


# Helper function to convert qualifying time from "minute:second.millisecond" format into total seconds
def stringTimetoInt(time):
    try:
        minutes, rest = time.split(':')
    except ValueError:
        # If a value error has occurred, then the issue is that .split() returned one argument instead of two
        # This only happens when the input time is of the form "second.milliseconds" if the time was below a minute
        # Hence, set minutes to 0, and the rest of the string, which is of the form "second.milliseconds" as the original
        # input string
        minutes = 0
        rest = time
    seconds, milliseconds = rest.split('.')
    return (int(minutes) * 60) + int(seconds) + (int(milliseconds)/1000)

# Helper functions to convert any team name to standard name (so as to exclude engine/sponsorship/rebrand changes as
# being seperate new teams)
def updateTeamNames(team):
    try:
        return teamDict[team]
    except KeyError:
        # If key error, that means input team is not in the dictionary and so is in standard form already
        return team


# Dictionary containing translation to standard team names
teamDict = {
    "AlphaTauri Honda RBPT": "Racing Bulls",
    "Lotus Renault": "Alpine",
    "Toro Rosso Ferrari": "Racing Bulls",
    "Williams Toyota": "Williams",
    "Williams Cosworth": "Williams",
    "RBR Ferrari": "Red Bull",
    "Lotus Mercedes": "Alpine",
    "AlphaTauri Honda":	"Racing Bulls",
    "Alpine Renault": "Alpine",
    "McLaren Mercedes":	"McLaren",
    "RBR Renault": "Red Bull",
    "Red Bull Racing TAG Heuer": "Red Bull",
    "Renault": "Alpine",
    "Haas Ferrari": "Haas",
    "Lotus Cosworth": "Alpine",
    "Red Bull Racing RBPT": "Red Bull",
    "Scuderia Toro Rosso Honda": "Racing Bulls",
    "Kick Sauber Ferrari": "Sauber",
    "McLaren Renault": "McLaren",
    "Force India Mercedes": "Aston Martin",
    "Williams Mercedes": "Williams",
    "McLaren Honda": "McLaren",
    "Red Bull Racing Honda": "Red Bull",
    "Aston Martin Mercedes": "Aston Martin",
    "Toro Rosso": "Racing Bulls",
    "Alfa Romeo Ferrari": "Sauber",
    "Aston Martin Aramco Mercedes": "Aston Martin",
    "AlphaTauri RBPT": "Racing Bulls",
    "Red Bull Racing Honda RBPT": "Red Bull",
    "RB Honda RBPT": "Red Bull",
    "Sauber BMW": "Sauber",
    "STR Ferrari": "STR",
    "Red Bull Racing Renault": "Red Bull",
    "STR Renault": "STR",
    "STR Cosworth":	"STR",
    "Racing Point BWT Mercedes": "Aston Martin",
    "Alfa Romeo Racing Ferrari": "Sauber",
    "Red Bull Renault": "Red Bull",
    "Williams Renault":	"Williams",
    "Force India Ferrari": "Aston Martin",
    "Sauber Ferrari": "Sauber"
}

# Helper function to find all input columns for model training (especially one-hot encoded ones)
def findInputColumns(df):
    #inputColumns = ["Position Quali", "Last 5 Race", "Quali Delta", "Team Season DNF", "Last 5 DNF", "SPosition"]
    inputColumns = ["Position Quali", "SPosition", "Team Season DNF",  "Last 5 Race", "Last 5 Points", "Last 5 DNF", "Quali Delta", "WDC", "WCC"]
    for column in df.columns:
        if column.startswith("Driver_") or column.startswith("Team_") or column.startswith("Race_"):
            inputColumns.append(column)
    # inputColumns = df.columns
    # inputColumns.drop(["Year", "Q3", "Position Race", "Points", "Win", "Race DNF", "Race Number", "Season Points", "Fastest Individual Time", "Fastest Quali Time"])
    return inputColumns

# Helper function to find amount of correctly predicted wins for a regression model
def regressionPrecision(testdf,predictions):
    # Create new dataframe with actual result and predicted results
    predicteddf = pd.DataFrame({"Actual": testdf["Position Race"], "Predicted": predictions, "Race": testdf["Race Number"]})
    # Find number of races in the dataframe
    numberOfRaces = predicteddf["Race"].max() - predicteddf["Race"].min() + 1
    # Sort the entries by race and then by the predicted position of the regression model within that race
    sorteddf = predicteddf.sort_values(by=["Race", "Predicted"], ascending=[True, True])
    # Find the row for each race with the lowest predicted position (i.e. the race winner)
    sorteddf["First"] = sorteddf.groupby("Race").cumcount() == 0
    # Filter the rows to only where the prediction of the race winner was correct
    correctdf = sorteddf[(sorteddf["First"]) & (sorteddf["Actual"] == 1)]
    # Find the number of correct predictions (number of rows in correct dataframe)
    correct = correctdf.shape[0]
    return (correct/numberOfRaces)


# Function that trains a random forest regression model
def randomForest(train, test, inputColumns):
    # Train model
    model = RandomForestRegressor(n_estimators=1000, min_samples_split=100, random_state=1, n_jobs=-1)
    model.fit(train[inputColumns], train["Position Race"])
    # Make predictions on testing set
    predictions = model.predict(test[inputColumns])
    print("Random Forest:", regressionPrecision(test, predictions))


# Function that trains a linear regression model
def linearRegression(train, test, inputColumns):
    # Train model
    model = LinearRegression(fit_intercept=False, n_jobs=-1)
    model.fit(train[inputColumns], train["Position Race"])
    # Make predictions on testing set
    predictions = model.predict(test[inputColumns])
    print("Linear Regression:", regressionPrecision(test, predictions))


# Function that trains a logistic regression model
def logisticRegression(train, test, inputColumns):
    # Train model
    model = LogisticRegression(penalty='l2',
                               C=1,
                               solver='saga',
                               max_iter=10000,
                               n_jobs=-1)
    model.fit(train[inputColumns], train["Win"])
    # Make predictions on testing set
    predictions = model.predict_proba(test[inputColumns])
    prob1 = []
    for element in predictions:
        prob1.append(element[0])
    print("Logistic Regression:", regressionPrecision(test, prob1))


def machineLearningTraining(df):
    # Split the dataset into training set (80%) and testing set (20%)
    train = df[df["Year"] < 2021]
    test = df[df["Year"] >= 2021]
    # Find all input columns for the models
    inputColumns = findInputColumns(df)
    randomForest(train, test, inputColumns)
    linearRegression(train, test, inputColumns)
    logisticRegression(train, test, inputColumns)
    return


def main():
    df = dataProcessing()
    machineLearningTraining(df)

main()

