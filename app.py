import streamlit as st
import xgboost as xgb
import pandas as pd

# Load model
model = xgb.Booster()
model.load_model("xgboost_home_model.json")

st.title("🏠 Home Price Predictor")

# Collect inputs
st.header("Enter Property Details")

# Steps
st.write("Feel free to edit the information. Be sure to put an actual Sacramento City and Zip Code")
st.write("This Linear Regression model from XGBoost uses 53,000 sold homes data since the end of 2021 in Sacramento County alone")
st.write("Rsquared = 88.9%")

# Define form inputs
user_inputs = {
    "ZipCode": st.text_input("ZIP Code", "95825"),
    "City": st.text_input("City", "Sacramento"),
    "Bedrooms": st.number_input("Bedrooms", 0, 10, 3),
    "PossibleBedrooms": st.number_input("Possible Bedrooms", 0, 5, 0),
    "FullBaths": st.number_input("Full Baths", 0, 5, 2),
    "HalfBaths": st.number_input("Half Baths", 0, 3, 1),
    "SqFt": st.number_input("Square Footage", 500, 10000, 2000),
    "LotSize": st.number_input("Lot Size", 500, 20000, 5000),
    "GarageSpaces": st.number_input("Garage Spaces", 0, 5, 2),
    "YearBuilt": st.number_input("Year Built", 1900, 2025, 1990),
    "DaysOnMarket": st.number_input("Days on Market", 0, 365, 30),
    "Fireplaces": st.number_input("Fireplaces", 0, 5, 1),
    "CloseMonth": st.number_input("Close Month (1-12)", 1, 12, 5),
    "CloseYear": st.number_input("Close Year, enter 2025", 2025, 2025, 2025),
    "TotalPotentialBedrooms": st.number_input("Total Potential Bedrooms", 0, 10, 3),

    # Binary features
    "CoolingCentral": int(st.checkbox("Has Central Cooling")),
    "HeatingCentral": int(st.checkbox("Has Central Heating")),
    "HasPool": int(st.checkbox("Has Pool")),
    "HasNewKitchen": int(st.checkbox("New Kitchen")),
    "HasNewRoof": int(st.checkbox("New Roof")),
    "HasNewBathroom": int(st.checkbox("New Bathroom")),
    "HasRemodel": int(st.checkbox("Recently Remodeled")),
    "MentionsLuxury": int(st.checkbox("Mentions Luxury")),
    "MentionsView": int(st.checkbox("Mentions View")),
    "HasRiverfront": int(st.checkbox("River/Lake/Waterfront")),
    "IsFixerUpper": int(st.checkbox("Is Fixer Upper")),
    "HasSolar": int(st.checkbox("Has Solar")),
    "HasWholeHouseFan": int(st.checkbox("Whole House Fan")),
    "HasUpdatedFlooring": int(st.checkbox("Updated Flooring")),
    "HasUpdatedKitchen": int(st.checkbox("Updated Kitchen")),
    "HasUpdatedBathrooms": int(st.checkbox("Updated Bathrooms")),
    "HasBackyardRetreat": int(st.checkbox("Backyard Retreat")),
    "HasADU": int(st.checkbox("Has ADU/Shed")),
    "HasRVAccess": int(st.checkbox("RV Access"))
}

# Prediction
if st.button("Predict Price"):
    df = pd.DataFrame([user_inputs])

    # Match model expectations
    df["ZipCode"] = df["ZipCode"].astype("category")
    df["City"] = df["City"].astype("category")
    df["CloseMonth"] = df["CloseMonth"].astype("category")

    # Use same order as model
    feature_order = [
        "ZipCode", "Bedrooms", "FullBaths", "HalfBaths", "SqFt", "LotSize",
        "YearBuilt", "DaysOnMarket", "GarageSpaces", "CoolingCentral", "HeatingCentral",
        "HasPool", "CloseMonth", "HasNewKitchen", "HasNewRoof", "HasRemodel",
        "HasRiverfront", "MentionsLuxury", "MentionsView", "HasNewBathroom",
        "PossibleBedrooms", "TotalPotentialBedrooms", "CloseYear", "IsFixerUpper",
        "Fireplaces", "HasSolar", "HasWholeHouseFan", "HasUpdatedFlooring",
        "HasUpdatedKitchen", "HasUpdatedBathrooms", "HasBackyardRetreat", "HasADU",
        "HasRVAccess", "City"
    ]
    df = df[feature_order]

    dinput = xgb.DMatrix(df, enable_categorical=True)
    price = model.predict(dinput)[0]

    st.success(f"🏡 Predicted Home Price: ${price:,.0f}")