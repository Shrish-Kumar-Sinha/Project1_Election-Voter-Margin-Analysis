#Importing necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import lognorm
from scipy.stats import chisquare

eci_2026 = pd.read_excel(
    "eci_2026_final_data.xlsx",
    sheet_name="2026_eci_winners_appended"
)
print(eci_2026.shape)

eci_2026.head()

def test_lognorm_fit(
        states=None,
        metric="WINNING MARGIN",
        bins=10):

    # ------------------------------------
    # Filter Data
    # ------------------------------------

    if states is None:

        filtered_df = eci_2026.copy()

        state_label = "All States"

    else:

        if isinstance(states, str):
            states = [states]

        filtered_df = eci_2026.loc[
            eci_2026["STATE/UT"].isin(states)
        ]

        state_label = ", ".join(states)

    data = filtered_df[metric].dropna()

    # Lognormal requires positive values
    data = data[data > 0]

    if len(data) < 10:

        print(
            "Too few observations for reliable fitting."
        )

        return None

    # ------------------------------------
    # Fit Lognormal Distribution
    # ------------------------------------

    shape, loc, scale = lognorm.fit(data)

    # ------------------------------------
    # Create Figure
    # ------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    observed_freq, bin_edges, _ = ax.hist(
        data,
        bins=bins,
        alpha=0.6,
        label="Observed Data"
    )

    # ------------------------------------
    # Expected Curve
    # ------------------------------------

    x_values = np.linspace(
        data.min(),
        data.max(),
        500
    )

    fitted_pdf = lognorm.pdf(
        x_values,
        shape,
        loc,
        scale
    )

    bin_width = (
        bin_edges[1]
        - bin_edges[0]
    )

    expected_frequency_curve = (
        fitted_pdf
        * len(data)
        * bin_width
    )

    ax.plot(
        x_values,
        expected_frequency_curve,
        linewidth=2,
        color="red",
        label="Fitted Lognormal"
    )

    ax.set_title(
        f"{metric} Distribution - {state_label}"
    )

    ax.set_xlabel(metric)

    ax.set_ylabel("Frequency")

    ax.legend()

    ax.grid(True)

    # ------------------------------------
    # Chi-Square Test
    # ------------------------------------

    cdf_upper = lognorm.cdf(
        bin_edges[1:],
        shape,
        loc,
        scale
    )

    cdf_lower = lognorm.cdf(
        bin_edges[:-1],
        shape,
        loc,
        scale
    )

    expected_prob = (
        cdf_upper
        - cdf_lower
    )

    expected_freq = (
        expected_prob
        * len(data)
    )

    expected_freq = (
        expected_freq
        * observed_freq.sum()
        / expected_freq.sum()
    )

    chi2_stat, p_value = chisquare(
        observed_freq,
        expected_freq
    )

    alpha = 0.05

    if p_value > alpha:

        conclusion = (
            "Log-normal distribution is a good fit to the data."
        )

    else:

        conclusion = (
            "Log-normal distribution is not a good fit to the data."
        )

    # ------------------------------------
    # Display Results
    # ------------------------------------

    plt.close(fig)

        # ------------------------------------
    # Return Results
    # ------------------------------------

    return (
        fig,
        conclusion,
        f"""
States: {state_label}

Constituencies: {len(data)}

Chi-Square Statistic: {chi2_stat:.4f}

P-value: {p_value:.4f}
"""
    )

def run_analysis(selected_states):

    if not selected_states:

        selected_states = None

    fig, interpretation, stats = test_lognorm_fit(
        states=selected_states
    )

    return fig, interpretation, stats

state_list = sorted(
    eci_2026["STATE/UT"].unique()
)

import gradio as gr

demo = gr.Interface(
    fn=run_analysis,

    inputs=gr.CheckboxGroup(
        choices=state_list,
        label="Select State(s)"
    ),

    outputs=[
        gr.Plot(label="Distribution"),
        gr.Textbox(
            label="Interpretation",
            lines=8
        ),
        gr.Textbox(
            label="Statistical Details",
            lines=8
        )
    ],

    title="Election Behaviour Analysis",

    description="""
Select one or more states to analyse
the distribution of winning margins.
"""
)

demo.launch()