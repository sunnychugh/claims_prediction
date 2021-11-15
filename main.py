import argparse
import config
import regression


def parse_cmdline_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Hastings Direct claims prediction test..."
    )
    parser.add_argument(
        "-r",
        "--regressor",
        choices=[
            "XGBRegressor",
            "RandomForestRegressor",
            "GradientBoostingRegressor",
            "SVR",
        ],
        default=config.MODEL,
        help="Regressor to fit the data",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Increase verbosity"
    )
    return parser.parse_args()


def main():
    args = parse_cmdline_arguments()
    regr = regression.Regression(args)

    # Read the claims data
    df_claims = regr.read_excel(config.DATAFILE)

    # Preprocess data (remove NaN, grouped data, etc)
    num_inputs = len(config.COLS) - 1
    df_claims = regr.preprocess_data(df_claims, num_inputs)
    if args.verbose:
        print(df_claims)

    # Divide data in train and test/validation set
    X_train, X_test, y_train, y_test = regr.split_data(df_claims, num_inputs)

    # Train/Fit the model on the training data
    if args.regressor == "XGBRegressor":
        regr_model = regr.fit_XGBRegressor(X_train, y_train, X_test, y_test)
    elif args.regressor == "RandomForestRegressor":
        regr_model = regr.fit_RandomForestRegressor(X_train, y_train)
    elif args.regressor == "GradientBoostingRegressor":
        regr_model = regr.fit_GradientBoostingRegressor(X_train, y_train)
    elif args.regressor == "SVR":
        regr_model = regr.fit_SVR(X_train, y_train)

    # Display feature importance value of each data column
    if args.regressor != "SVR":
        regr.print_feature_importances(regr_model)

    # Make predictions on test set
    mse, score = regr.predict(regr_model, X_test, y_test)
    if args.verbose:
        print(mse, score)


if __name__ == "__main__":
    main()
