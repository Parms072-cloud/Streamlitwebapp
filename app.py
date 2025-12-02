# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# from datetime import datetime, timedelta

# st.set_page_config(page_title="Time to Next Service Predictor", layout="centered")

# # ========= LOAD ARTIFACTS =========
# @st.cache_resource
# def load_artifacts():
#     model = joblib.load("rf_ttns_model.joblib")        # matches your file
#     feature_cols = joblib.load("feature_columns.joblib")  # matches your file

#     # encoders file must be created in Colab – see below
#     encoders = joblib.load("ttns_encoders.joblib")

#     return model, encoders, feature_cols

# model, encoders, feature_cols = load_artifacts()

# cat_cols = ["make", "model", "dealer_name", "brand_match"]

# # We can pull allowed categories from encoders
# make_options = list(encoders["make"].classes_)
# dealer_options = list(encoders["dealer_name"].classes_)
# model_options = list(encoders["model"].classes_)  # if too many, you can change to text input
# brand_match_options = list(encoders["brand_match"].classes_)

# # ========= APP UI =========
# st.title("🚗 Time to Next Service (TTNS) Predictor")

# st.write(
#     "This app uses a trained regression model to estimate how long until a vehicle "
#     "is likely to return for service, based on historical Go Auto patterns."
# )

# st.sidebar.header("Input Vehicle & Service Details")

# # --- Inputs ---

# selected_make = st.sidebar.selectbox("Vehicle Make", make_options)
# selected_model = st.sidebar.selectbox("Vehicle Model", model_options)
# mileage = st.sidebar.number_input("Current Mileage (km)", min_value=0, max_value=500000, value=60000, step=1000)
# selected_dealer = st.sidebar.selectbox("Servicing Dealer", dealer_options)

# brand_match_display = st.sidebar.selectbox(
#     "Brand Match (is this the same brand as the dealer?)",
#     brand_match_options
# )

# service_month = st.sidebar.number_input("Service Month", min_value=1, max_value=12, value=datetime.today().month)
# service_year = st.sidebar.number_input("Service Year", min_value=2000, max_value=2100, value=datetime.today().year)

# # Auto-calc is_luxury from make
# luxury_makes = ["BMW", "LEXUS", "MERCEDES", "JAGUAR", "AUDI", "PORSCHE", "VOLVO", "CADILLAC", "INFINITY"]
# is_luxury = 1 if selected_make in luxury_makes else 0

# st.sidebar.write(f"**Luxury Flag (auto):** {is_luxury}")

# # ========= BUILD FEATURE ROW =========

# def build_feature_row():
#     # Encode categorical features with same LabelEncoders
#     encoded_make = encoders["make"].transform([selected_make])[0]
#     encoded_model = encoders["model"].transform([selected_model])[0]
#     encoded_dealer = encoders["dealer_name"].transform([selected_dealer])[0]
#     encoded_brand_match = encoders["brand_match"].transform([brand_match_display])[0]

#     row = {
#         "make": encoded_make,
#         "model": encoded_model,
#         "mileage": mileage,
#         "dealer_name": encoded_dealer,
#         "is_luxury": is_luxury,
#         "brand_match": encoded_brand_match,
#         "service_month": service_month,
#         "service_year": service_year,
#     }

#     df = pd.DataFrame([row])

#     # Ensure column order matches training
#     df = df.reindex(columns=feature_cols, fill_value=0)

#     return df

# # ========= PREDICTION =========

# if st.button("Predict Time to Next Service"):
#     try:
#         input_df = build_feature_row()
#         pred_days = model.predict(input_df)[0]

#         # Clip negative predictions to 0
#         pred_days = max(pred_days, 0)

#         st.subheader("Prediction Result")
#         st.metric("Estimated Time to Next Service", f"{pred_days:.1f} days")

#         est_date = datetime.today() + timedelta(days=pred_days)
#         st.write(f"📅 **Estimated service date:** {est_date.strftime('%Y-%m-%d')}")

#         # Some business interpretation
#         if is_luxury:
#             st.info(
#                 "This is classified as a **luxury** vehicle. "
#                 "The dealership can use this estimate to send premium, earlier reminders "
#                 "and allocate experienced technicians."
#             )
#         else:
#             st.info(
#                 "This is a **non-luxury** vehicle. "
#                 "Use this estimate for standard reminder cycles and staffing planning."
#             )

#         st.caption("Note: This is a statistical prediction based on historical patterns, not a guarantee.")

#     except Exception as e:
#         st.error(f"Prediction failed: {e}")

import streamlit as st
import joblib
import pandas as pd
import os

st.set_page_config(page_title="TTNS DEBUG APP", layout="centered")

st.title("🔧 TTNS Debug ParminderApp")

st.write("If you can see this text, the Streamlit UI is working.")

st.subheader("1. Files in current folder")
st.write(os.listdir("."))

st.subheader("2. Test loading model + feature columns")

if st.button("Test load artifacts"):
    try:
        model = joblib.load("rf_ttns_model.joblib")          # uses your actual file name
        feature_cols = joblib.load("feature_columns.joblib") # uses your actual file name

        st.success("✅ Loaded model and feature_columns.joblib successfully!")
        st.write("Number of features:", len(feature_cols))
        st.write("Feature names:", feature_cols)

    except Exception as e:
        st.error("❌ Failed to load artifacts:")
        st.code(str(e))

