import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================
# PAGE SETTINGS
# ============================================

st.set_page_config(
    page_title="Manufacturing Defect Prediction",
    page_icon="🏭",
    layout="wide"
)


# ============================================
# LOAD SAVED MODEL
# ============================================

@st.cache_resource
def load_model():

    pipeline = joblib.load(
        "final_defect_prediction_pipeline.pkl"
    )

    selected_features = joblib.load(
        "selected_features.pkl"
    )

    return pipeline, selected_features


pipeline, selected_features = load_model()


# ============================================
# TITLE
# ============================================

st.title("🏭 Manufacturing Defect Prediction")

st.write(
    "Statistical Process Improvement & Defect Reduction Analysis"
)

st.info(
    "Upload manufacturing process measurements "
    "to identify records that may require inspection."
)


# ============================================
# SIDEBAR
# ============================================

st.sidebar.header("Project Information")

st.sidebar.write(
    "**Model:** Logistic Regression"
)

st.sidebar.write(
    "**Selected Measurements:** 50"
)

st.sidebar.write(
    "**Dataset:** Manufacturing Process Data"
)


# ============================================
# FILE UPLOAD
# ============================================

st.header("Upload Manufacturing Data")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)


# ============================================
# PROCESS UPLOADED DATA
# ============================================

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Total Records",
            df.shape[0]
        )

    with col2:
        st.metric(
            "Total Columns",
            df.shape[1]
        )

    st.dataframe(
        df.head()
    )


    # ========================================
    # CHECK REQUIRED FEATURES
    # ========================================

    missing_features = [
        feature
        for feature in selected_features
        if feature not in df.columns
    ]


    if len(missing_features) > 0:

        st.error(
            "Some required measurements are missing "
            "from the uploaded CSV."
        )

        st.write(
            "Missing measurements:"
        )

        st.write(
            missing_features
        )


    else:

        # ====================================
        # SELECT REQUIRED MEASUREMENTS
        # ====================================

        X = df[selected_features].copy()


        # ====================================
        # HANDLE MISSING VALUES
        # ====================================

        X = X.fillna(
            X.median()
        )


        # ====================================
        # MAKE PREDICTIONS
        # ====================================

        predictions = pipeline.predict(X)

        probabilities = (
            pipeline.predict_proba(X)[:, 1]
        )


        # ====================================
        # ADD RESULTS
        # ====================================

        results = df.copy()

        results["Defect_Probability"] = (
            probabilities
        )

        results["Prediction"] = np.where(
            predictions == 1,
            "Potential Defect",
            "Good"
        )


        # ====================================
        # SUMMARY
        # ====================================

        total_records = len(results)

        good_records = (
            predictions == 0
        ).sum()

        potential_defects = (
            predictions == 1
        ).sum()


        st.subheader(
            "Prediction Summary"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Total Records",
                total_records
            )


        with col2:

            st.metric(
                "Good Records",
                good_records
            )


        with col3:

            st.metric(
                "Potential Defects",
                potential_defects
            )


        # ====================================
        # RESULTS
        # ====================================

        st.subheader(
            "Prediction Results"
        )

        st.dataframe(
            results
        )


        # ====================================
        # FLAGGED RECORDS
        # ====================================

        st.subheader(
            "Records Requiring Inspection"
        )

        defect_results = results[
            results["Prediction"] ==
            "Potential Defect"
        ]


        if len(defect_results) > 0:

            st.warning(
                f"{len(defect_results)} records "
                "were flagged for inspection."
            )

            st.dataframe(
                defect_results
            )

        else:

            st.success(
                "No potential defects detected."
            )


        # ====================================
        # DOWNLOAD RESULTS
        # ====================================

        csv = results.to_csv(
            index=False
        )

        st.download_button(
            label="⬇️ Download Prediction Results",
            data=csv,
            file_name="defect_predictions.csv",
            mime="text/csv"
        )


else:

    st.warning(
        "Please upload a CSV file to begin prediction."
    )