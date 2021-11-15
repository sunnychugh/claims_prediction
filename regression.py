import pandas as pd
import xgboost
from sklearn import svm
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import config


class Regression:
    def __init__(self, args):
        self.args = args

    def read_excel(self, datafile):
        df = pd.read_excel(datafile["file"], sheet_name=datafile["sheetname"])
        return df

    def preprocess_data(self, df, num_inputs):
        # Drop the NA values
        df = df.dropna()
        # Feature encoding
        # Categorical boolean mask
        categorical_feature_mask = df.dtypes == object
        # Filter categorical columns using mask and turn it into a list
        categorical_cols = df.columns[categorical_feature_mask].tolist()
        # Instantiate labelencoder object
        le = LabelEncoder()
        df[categorical_cols] = df[categorical_cols].apply(
            lambda col: le.fit_transform(col)
        )
        # Select only the specified columns (features reduction)
        df = df[config.COLS]
        # Drop the rows with Incurred columns values = zero
        df = df[df["Incurred"] != 0]
        # Take the mean of same columns entries
        grouped = df.groupby(config.COLS[:(num_inputs)])
        df = grouped.mean().reset_index()
        return df

    def split_data(self, df_claims, num_inputs):
        df_claims = df_claims.to_numpy()
        X = df_claims[:, range(0, num_inputs)]
        y = df_claims[:, range(num_inputs, num_inputs + 1)]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True
        )
        return X_train, X_test, y_train, y_test

    def fit_XGBRegressor(self, X_train, y_train, X_test, y_test):
        print("\nRegressor selected: XGBRegressor")
        regr = xgboost.XGBRegressor()
        regr.fit(X_train, y_train)
        # Define regression model with more parameters
        # regr = xgboost.XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
        # regr.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        return regr

    def fit_RandomForestRegressor(self, X_train, y_train):
        print("\nRegressor selected: RandomForestRegressor")
        regr = RandomForestRegressor(max_depth=20, random_state=0)
        regr.fit(X_train, y_train)
        return regr

    def fit_GradientBoostingRegressor(self, X_train, y_train):
        print("\nRegressor selected: GradientBoostingRegressor")
        regr = GradientBoostingRegressor(max_depth=2, n_estimators=120)
        regr.fit(X_train, y_train)
        return regr

    def fit_SVR(self, X_train, y_train):
        print("\nRegressor selected: SVR")
        regr = svm.SVR()
        regr.fit(X_train, y_train)
        return regr

    def print_feature_importances(self, regr_model):
        for name, score in zip(config.COLS[:-1], regr_model.feature_importances_):
            print(name, score)

    def predict(self, regr_model, X_test, y_test):
        y_pred = regr_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        # score: Return the coefficient of determination of the prediction
        score = regr_model.score(X_test, y_test)
        print("y_test:\n", y_test.reshape(1, -1), "\ny_pred:\n", y_pred)
        return mse, score
