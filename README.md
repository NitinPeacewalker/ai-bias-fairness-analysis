# AI Bias & Fairness Analysis

## Overview

This project investigates algorithmic bias and fairness in a machine learning classification model.

The objective is to understand whether a model produces different outcomes across demographic groups and to evaluate its performance using quantitative fairness metrics.

## Research Question

> Does the machine learning model produce systematically different outcomes across demographic groups?

## Dataset

The analysis uses the India Human Development

Survey-II (IHDS-II), 2011-12 dataset obtained from The Data Sharing for Demographic Research (DSDR)

**Dataset source:** [https://www.icpsr.umich.edu/web/DSDR/studies/36151/versions/V6#](https://www.icpsr.umich.edu/web/DSDR/studies/36151/versions/V6#)

The dataset contains information related toThe India Human Development Survey-II (IHDS-II), 2011-12 is a nationally representative, multi-topic survey of 42,152 households in 1,420 villages and 1,042 urban neighborhoods across India. These data are mostly re-interviews of households interviewed for IHDS (ICPSR 22626) in 2004-05. Both surveys cover all states and union territories of India with the exception of Andaman & Nicobar and Lakshadweep. Two one-hour interviews in each household covered topics concerning health, education, employment, economic status, marriage, fertility, gender relations, social capital, village infrastructure, wage levels, and panchayat composition. Childrenaged 8-11 completed short reading, writing and arithmetic tests. 

### Model features

- Age
- Education
- Rural/Urban residence

### Protected attributes

- Sex
- Caste/Social Group
- Religion

### Target variable

The model predicts employed

Protected attributes were not directly used as predictors in the baseline Logistic Regression model. They were instead used to evaluate whether model outcomes differed across demographic groups.

## Data Preparation

The dataset was cleaned and prepared before model training.

The preprocessing included:

- Handling missing values
- Encoding categorical variables
- Preparing numerical variables
- Defining the target variable
- Splitting the dataset into training and test sets
- Preparing demographic attributes for fairness evaluation

The exact preprocessing steps are documented in the Jupyter Notebook.

## Model

A Logistic Regression classifier was used as the baseline model.

The model uses:

- Age
- Education
- Rural/Urban residence

Protected attributes such as sex, caste/social group, and religion were not directly used as model predictors.

## Fairness Analysis

The model was evaluated using fairness metrics including:

- **Demographic Parity Difference**
- **Equal Opportunity Difference**

The analysis examined differences across:

- Sex
- Caste/Social Group
- Religion

These metrics were selected to examine different dimensions of fairness rather than relying only on overall model accuracy.

## Results

| Protected Attribute        | Demographic Parity Difference | Equal Opportunity Difference |
| -------------------------- | ----------------------------: | ---------------------------: |
| Sex                        |                        0.0152 |                       0.1639 |
| Caste/Social Group         |                        0.1363 |                       0.1184 |
| Religion (groups N >= 100) |                        0.1614 |                       0.2355 |

### Interpretation

The results show that fairness outcomes vary depending on the protected attribute and fairness criterion.

For example, the demographic parity difference for sex is relatively small (0.0152), while the equal opportunity difference is larger (0.1639).

For religion, the equal opportunity difference is 0.2355, indicating a larger disparity in the model's true-positive rates between the evaluated groups.

These results demonstrate why evaluating only overall model performance is insufficient when assessing algorithmic fairness.

## Methodology

1. Load and inspect the dataset
2. Identify relevant demographic attributes
3. Clean and preprocess the data
4. Define model features and target variable
5. Train a Logistic Regression baseline
6. Generate predictions
7. Evaluate model performance
8. Calculate fairness metrics
9. Compare outcomes across demographic groups
10. Interpret potential disparities
11. Document limitations and possible mitigation approaches

## Tools & Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Fairlearn
- Jupyter Notebook
- Matplotlib

## Limitations

This analysis should not be interpreted as proof that the model is unbiased or biased in a real-world deployment.

Fairness metrics depend on the dataset, target definition, model, protected-group definitions, sample sizes, and fairness criterion being used.

Some demographic groups may have relatively small sample sizes. Results for small groups should therefore be interpreted cautiously.

The analysis identifies statistical disparities but does not by itself establish the social or causal reasons behind those disparities.

Further analysis is required to investigate the causes of observed disparities.

## Future Work

- Compare multiple machine learning models
- Add additional fairness metrics
- Perform intersectional fairness analysis
- Investigate class imbalance
- Test bias mitigation techniques
- Compare models before and after fairness intervention
- Evaluate calibration across demographic groups
- Investigate whether disparities persist across different datasets

## Reproducibility

The analysis is provided as a Jupyter Notebook.

To reproduce the analysis:

```bash
pip install -r requirements.txt
```

Then open:

```text
notebooks/fairness_analysis.ipynb
```

## Author

**Nitin Sonawane**

Interested in Responsible AI, AI Safety, AI Governance, algorithmic fairness, and ethical applications of machine learning.
