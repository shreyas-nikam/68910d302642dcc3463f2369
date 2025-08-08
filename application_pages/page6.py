import streamlit as st
import pandas as pd
from sklearn.metrics import mean_absolute_error
import plotly.express as px

def run_page6():
    st.header("Model Evaluation")
    st.markdown("""
    This section allows for evaluating the performance of the LGD models (TTC and PIT). 
    We'll focus on Mean Absolute Error (MAE) as the primary metric, 
    visualizing the distribution of errors, and comparing TTC vs. PIT model performance.
    """)

    # Check if necessary data is available in st.session_state
    if "processed_loan_data" not in st.session_state or st.session_state["processed_loan_data"] is None:
        st.warning("Please complete Feature Engineering first.")
        return

    if "ttc_model" not in st.session_state or st.session_state["ttc_model"] is None:
        st.warning("Please train the TTC model first.")
        return

    if "pit_overlay_model" not in st.session_state or st.session_state["pit_overlay_model"] is None:
        st.info("PIT Overlay model not trained. Evaluating TTC model only.")

    # Load data from session state
    df = st.session_state["processed_loan_data"].copy()

    # --- TTC Model Evaluation --- #
    st.subheader("TTC Model Evaluation")
    if "ttc_predictions_train" in st.session_state and st.session_state["ttc_predictions_train"] is not None:
        ttc_predictions_train = st.session_state["ttc_predictions_train"]
        y_train = df["LGD_realised"] # Assuming all data was used for training in TTC (for simplicity)

        mae_ttc_train = mean_absolute_error(y_train, ttc_predictions_train)
        st.metric(label="Mean Absolute Error (TTC - Training Data)", value=f"{mae_ttc_train:.4f}")

        # Error distribution plot
        errors_ttc = y_train - ttc_predictions_train
        fig_error_dist_ttc = px.histogram(errors_ttc, nbins=50, title="Distribution of TTC Model Errors (Training Data)",
                                       labels={"value": "Error (Actual - Predicted)"},
                                       template="plotly_white")
        st.plotly_chart(fig_error_dist_ttc, use_container_width=True)
    else:
        st.info("TTC model predictions not found. Please train the TTC model and ensure predictions are generated.")

    # --- PIT Model Evaluation --- #
    st.subheader("PIT Model Evaluation")
    if "pit_overlay_model" in st.session_state and st.session_state["pit_overlay_model"] is not None and \
       "aligned_lgd_macro_data" in st.session_state and st.session_state["aligned_lgd_macro_data"] is not None:

        aligned_data = st.session_state["aligned_lgd_macro_data"].copy()

        if 'pit_lgd_predicted' in aligned_data.columns:
            y_actual_pit = aligned_data["mean_realized_lgd"]  # Use aggregated realized LGD
            y_predicted_pit = aligned_data["pit_lgd_predicted"] # Aggregated PIT predictions

            mae_pit = mean_absolute_error(y_actual_pit, y_predicted_pit)
            st.metric(label="Mean Absolute Error (PIT - Aggregated Data)", value=f"{mae_pit:.4f}")

            # Error distribution plot for PIT model
            errors_pit = y_actual_pit - y_predicted_pit
            fig_error_dist_pit = px.histogram(errors_pit, nbins=50, title="Distribution of PIT Model Errors (Aggregated Data)",
                                           labels={"value": "Error (Actual - Predicted)"},
                                           template="plotly_white")
            st.plotly_chart(fig_error_dist_pit, use_container_width=True)

            # TTC vs PIT Comparison
            st.subheader("TTC vs PIT Comparison")
            st.markdown("This chart compares the Mean Absolute Error for the TTC and PIT models, providing insight into their relative performance.")

            mae_data = pd.DataFrame({
                "Model": ["TTC", "PIT"],
                "MAE": [mae_ttc_train, mae_pit] # Use the calculated MAEs
            })
            fig_mae_comparison = px.bar(mae_data, x="Model", y="MAE", title="MAE Comparison: TTC vs PIT",
                                        color="Model", labels={"MAE": "Mean Absolute Error"},
                                        template="plotly_white")
            st.plotly_chart(fig_mae_comparison, use_container_width=True)

        else:
            st.warning("PIT LGD predictions not found in aligned data.")

    else:
        st.info("Train the PIT overlay model and generate predictions to evaluate its performance.")

    st.markdown("""
    --- 
    **Summary of Model Evaluation:**
    *   MAE provides a simple and interpretable measure of the average prediction error.
    *   Comparing the error distributions of the TTC and PIT models helps understand their biases.
    *   The PIT overlay aims to improve upon the TTC model, particularly in capturing the impact of macroeconomic factors.
    """)
