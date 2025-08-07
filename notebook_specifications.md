
## Jupyter Notebook Specification: LGD Model Development Lab

### 1. Notebook Overview

**Learning Goals:**

*   Understand the components of Loss Given Default (LGD) models, including Through-The-Cycle (TTC) and Point-In-Time (PIT) approaches.
*   Develop skills in data extraction, cleaning, and feature engineering for credit risk modeling.
*   Apply statistical modeling techniques to estimate LGD.
*   Learn how to incorporate macroeconomic factors into LGD models.
*   Evaluate model performance and prepare artifacts for deployment.

**Expected Outcomes:**

Upon completion of this lab, the user will be able to:

*   Calculate Realized LGD from loan-level data.
*   Perform exploratory data analysis to identify key LGD drivers.
*   Build TTC LGD models using appropriate regression techniques.
*   Create PIT overlays to adjust TTC LGDs based on macroeconomic conditions.
*   Assess model fit using relevant metrics and visualizations.
*   Prepare production-ready model artifacts.



### 2. Mathematical and Theoretical Foundations

**2.1 Realized LGD Calculation:**

The realized LGD is calculated for each defaulted loan as:

$$LGD_{realized} = \frac{EAD - PV(Recoveries) - PV(Collection\, Costs)}{EAD}$$

Where:

*   $LGD_{realized}$ is the realized Loss Given Default (a value between 0 and 1).
*   $EAD$ is the Exposure at Default, representing the outstanding principal at the time of default.
*   $PV(Recoveries)$ is the present value of all recoveries associated with the defaulted loan, discounted to the default date using the loan's effective interest rate.
*   $PV(Collection\, Costs)$ is the present value of all collection costs associated with the defaulted loan, discounted to the default date using the loan's effective interest rate.

The present value of a recovery payment at time $t$ is given by:

$$PV(Recovery_t) = \frac{Recovery_t}{(1 + r)^t}$$

where:

*   $Recovery_t$ is the recovery amount at time $t$ after default.
*   $r$ is the loan's effective interest rate.
*   $t$ is the time (in years) from the default date to the recovery payment date.

**Explanation:** Realized LGD represents the actual loss experienced by the lender when a borrower defaults. It considers the outstanding exposure at the time of default, any recoveries obtained, and the costs associated with the recovery process. Discounting future cashflows to present value is a standard practice in finance.

**Real-world application:** Accurately calculating realized LGD is crucial for financial institutions to estimate potential losses from their loan portfolios, set appropriate capital reserves, and comply with regulatory requirements.

**2.2 Through-The-Cycle (TTC) LGD Model:**

TTC LGD models estimate the long-run average LGD for a given segment, irrespective of current economic conditions. This is often modeled using Beta regression.

The Beta distribution is parameterized by two shape parameters, $\alpha$ and $\beta$, where the mean $\mu$ is given by:

$$\mu = \frac{\alpha}{\alpha + \beta}$$

The Beta regression model links the mean $\mu$ to a set of predictors using a link function $g(.)$, such as the logit link:

$$g(\mu) = X\beta$$

Where:

*   $\mu$ is the mean LGD to be predicted.
*   $X$ is the matrix of predictor variables.
*   $\beta$ are the coefficients to be estimated.
*   $g(.)$ is a link function (logit, probit, etc.) that maps $\mu$ (which is between 0 and 1) to the real number line.

Solving for $\mu$ given the logit link function:

$$\mu = \frac{e^{X\beta}}{1 + e^{X\beta}}$$

**Explanation:** Beta regression is suitable for modeling variables that are bounded between 0 and 1, such as LGD. The link function ensures that the predicted LGD values remain within the valid range. The predictors in the model can include loan characteristics, borrower attributes, and other relevant factors.

**Real-world application:** TTC LGD models provide a stable estimate of long-run average losses, which can be used for stress testing and regulatory capital calculations.

**2.3 Point-In-Time (PIT) LGD Overlay:**

PIT LGD models adjust the TTC LGD estimates to reflect current macroeconomic conditions. This is often achieved using linear regression.

$$LGD_{PIT} = LGD_{TTC} + \beta_1 \cdot Macroeconomic\, Factor_1 + \beta_2 \cdot Macroeconomic\, Factor_2 + ... + \epsilon$$

Where:

*   $LGD_{PIT}$ is the Point-In-Time LGD estimate.
*   $LGD_{TTC}$ is the Through-The-Cycle LGD estimate.
*   $Macroeconomic\, Factor_i$ is the value of the i-th macroeconomic indicator (e.g., unemployment rate, GDP growth).
*   $\beta_i$ is the coefficient for the i-th macroeconomic factor, estimated from historical data.
*   $\epsilon$ is the error term.

**Explanation:** PIT overlays capture the impact of macroeconomic conditions on LGD. By including macroeconomic factors in the model, the LGD estimates can be adjusted to reflect current or forecasted economic conditions.

**Real-world application:** PIT LGD models are used to assess the impact of economic downturns or upturns on potential losses, which can inform risk management decisions and capital planning.
