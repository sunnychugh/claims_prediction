# Specify the filenames containing the actual data
DATAFILE = {
    "file": "data/Data_Scientist_Interview_Task.xlsx",
    "sheetname": "Data",
}

# Define inputs and output data variables
COLS = ["Notifier", "Location_of_incident", "Weather_conditions", "Vehicle_mobile", "Main_driver", "PH_considered_TP_at_fault", "Incurred"]
# COLS = ["Location_of_incident","Weather_conditions", "Vehicle_mobile",  "Incurred"]

# Define the regressor model to be used
# Use 'XGBRegressor' or 'RandomForestRegressor' or 'GradientBoostingRegressor'
# or 'SVR' or 'DecisionTreeRegressor' or 'MLPRegressor'
MODEL = 'XGBRegressor'
