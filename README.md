Supervised Learning: Manual Linear Regression Optimization
Introduction

This Google Colab notebook demonstrates a fundamental concept in supervised learning: finding the optimal parameters for a linear regression model by iteratively minimizing a loss function. We use a simple 1D linear regression model to fit a line to a given set of input-output data points.
Objective

The primary objective of this notebook is to manually adjust the intercept (phi0) and slope (phi1) parameters of a linear regression model (y = phi0 + phi1 * x) to minimize the Mean Squared Error (MSE) loss function. The goal is to find the line that best fits the scattered data points, similar to Figure 2.2d in typical machine learning textbooks.
Dataset

The notebook utilizes a small, synthetic 1D dataset consisting of 12 input (x) and output (y) pairs. These points are designed to exhibit a general linear trend.

x = [0.03, 0.19, 0.34, 0.46, 0.78, 0.81, 1.08, 1.18, 1.39, 1.60, 1.65, 1.90] y = [0.67, 0.85, 1.05, 1.0, 1.40, 1.5, 1.3, 1.54, 1.55, 1.68, 1.73, 1.6 ]
Methodology

The optimization process is carried out through an iterative, coordinate-wise descent approach:

    Initialization: The model starts with initial phi0 and phi1 values (1.60 and -0.8 respectively).
    Loss Calculation: A compute_loss function calculates the Mean Squared Error (MSE) between the model's predictions and the actual y values.
    Iterative Adjustment:
        phi0 is adjusted while phi1 is kept constant, observing the change in loss.
        Then, phi1 is adjusted while phi0 is kept constant, again observing the loss.
        This process is repeated, guiding the parameters towards a minimum loss.
    Visualization: At each step, the model's line is plotted against the data points, and the current loss value is displayed, allowing for visual and quantitative assessment of the fit.
    Loss Landscape Visualization: Finally, a 2D heatmap of the loss function across a range of phi0 and phi1 values is generated, with the final optimized parameters marked, to visually demonstrate convergence to the minimum.

Key Findings

Through manual iterative adjustments, the notebook successfully identified the optimal parameters for the linear regression model:

    Final phi0 (Intercept): 0.6
    Final phi1 (Slope): 0.6
    Minimum Loss Achieved: 0.25

These parameters result in the line y = 0.6 + 0.6 * x, which provides an excellent visual fit to the data points and yields the lowest calculated loss.
How to Run the Notebook

    Open the notebook in Google Colab.
    Run all cells sequentially. The interactive process of adjusting phi0 and phi1 is demonstrated through multiple code cells and accompanying markdown explanations.

Dependencies

    numpy (for numerical operations)
    matplotlib (for plotting and visualization)
