"""
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import mean_absolute_error
import numpy as np

def run_model_evaluation():
    st.header("Model Evaluation")
    st.markdown("Evaluate model performance and visualize results.")

    if "y_test" not in st.session_state or "lgd_predictions_test_floored" not in st.session_state:
        st.warning("Please train the TTC model first in the 'TTC Model Building' section.")
        return

    y_test = st.session_state["y_test"]
    lgd_predictions_test_floored = st.session_state["lgd_predictions_test_floored"]

    st.subheader("Evaluation Metrics")
    mae = mean_absolute_error(y_test, lgd_predictions_test_floored)
    st.write(f"Mean Absolute Error (MAE): {mae:.4f}")

    st.subheader("Visualizations")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Predicted vs. Actual LGD")
        plot_df = pd.DataFrame({
            "Actual LGD": y_test,
            "Predicted LGD": lgd_predictions_test_floored
        })
        fig_scatter = px.scatter(plot_df, x="Actual LGD", y="Predicted LGD",
                                title="Predicted vs. Actual LGD (Test Set)",
                                labels={"Actual LGD": "Actual LGD", "Predicted LGD": "Predicted LGD"})
        fig_scatter.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="Red", dash="dash"), name="45-degree line")
        st.plotly_chart(fig_scatter)

    with col2:
        st.markdown("### Calibration Curve")
        bins = np.linspace(0, 1, 11) # 10 bins
        plot_df["Predicted_Bin"] = pd.cut(plot_df["Predicted LGD"], bins=bins, include_lowest=True)
        calibration_data = plot_df.groupby("Predicted_Bin").agg(
            mean_predicted=("Predicted LGD", "mean"),
            mean_actual=("Actual LGD", "mean")
        ).reset_index()
        calibration_data["Predicted_Bin_Mid"] = calibration_data["Predicted_Bin"].apply(lambda x: x.mid)

        fig_calibration = px.line(calibration_data, x="Predicted_Bin_Mid", y=["mean_predicted", "mean_actual"],
                                labels={"value": "LGD", "Predicted_Bin_Mid": "Mean Predicted LGD (Binned)"},
                                title="Calibration Curve",
                                line_dash_map={"mean_predicted": "dash", "mean_actual": "solid"})
        fig_calibration.update_layout(legend_title_text='Metric')
        st.plotly_chart(fig_calibration)

    st.subheader("Residuals vs. Fitted Plot")
    residuals = y_test - lgd_predictions_test_floored
    fig_residuals = px.scatter(x=lgd_predictions_test_floored, y=residuals,
                              title="Residuals vs. Fitted (Test Set)",
                              labels={"x": "Fitted LGD", "y": "Residuals"})
    fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_residuals)

"""