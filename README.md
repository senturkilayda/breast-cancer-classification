# Breast Cancer Classification: Logistic Regression from Scratch with NumPy

This project implements logistic regression from scratch using NumPy to classify breast tumors as benign or malignant.

The forward pass, sigmoid activation, binary cross-entropy loss, gradient computation, and weight updates are implemented manually to understand how the model learns.

The project started as an exercise in the MIUUL Deep Learning program. I extended it with a scikit-learn comparison, repeated experiments across 30 data splits, numerical gradient checking, decision threshold analysis, and visualizations.

## Why Logistic Regression?

Logistic regression can be viewed as a single neuron with a sigmoid activation function.

Its training process follows four steps:

1. Compute a weighted sum of the input features and add the bias.
2. Apply the sigmoid function to obtain probabilities.
3. Calculate the loss.
4. Update the weights and bias using gradient descent.

Implementing these steps with NumPy makes it possible to inspect the calculations and understand the learning process before moving to deeper neural networks.

## Dataset

The project uses the Wisconsin Diagnostic Breast Cancer dataset available in scikit-learn.

The dataset contains 569 samples and 30 numerical features extracted from digitized images of cell nuclei obtained through fine-needle aspiration.

The target labels are:

- `0`: Malignant
- `1`: Benign

There are 212 malignant and 357 benign samples.

Because malignant tumors are labeled `0`, malignant-class precision, recall, and F1-score are calculated using `pos_label=0`.

## Methodology

The data is split into 80% training and 20% test sets using `random_state=42`.

`StandardScaler` is fitted on the training data and then applied to the test data to avoid data leakage.

The NumPy model is trained with:

- Learning rate: `0.1`
- Training epochs: `100`
- Optimization: Full-batch gradient descent
- Loss function: Binary cross-entropy
- Prediction threshold: `0.5`

All 30 input features are used during training.

## Results

The NumPy implementation is compared with scikit-learn's `LogisticRegression` on the same train/test split.

| Metric | NumPy Implementation | scikit-learn LogisticRegression |
|---|---:|---:|
| Accuracy | 0.982 | 0.974 |
| Malignant Recall | 0.977 (42/43) | 0.953 (41/43) |
| ROC-AUC | 0.999 | 0.997 |

The scikit-learn model uses L2 regularization by default, while the NumPy implementation does not include regularization.

The accuracy difference corresponds to one test sample out of 114. This result is not enough to conclude that one implementation is generally better.

### How Reliable Is a Single Train/Test Split?

To examine how much the results depend on the data split, I repeated the experiment across 30 different random splits while preserving the class proportions.

The results show:

| Metric | Result |
|---|---:|
| Mean Accuracy | 0.978 |
| Accuracy Standard Deviation | 0.015 |
| Minimum Accuracy | 0.939 |
| Maximum Accuracy | 1.000 |
| Mean Malignant Recall | 0.956 |
| Lowest Malignant Recall | 0.857 |

The original test accuracy of 0.982 is slightly above the mean across the repeated experiments.

These results show why evaluating a model using only one random split can give an incomplete picture of its performance.

![Accuracy and malignant recall across 30 data splits](figures/fig4_bolme_degiskenligi.png)

### Error Analysis

The model makes two incorrect predictions on the original test split.

- A benign sample is classified as malignant with `P(Benign) = 0.4993`, just below the decision threshold.
- A malignant sample is classified as benign with `P(Benign) = 0.7192`. This prediction is farther from the decision boundary.

The two samples have similar mean radius, texture, perimeter, and area values. Their mean radius values are 13.34 and 13.80, respectively.

These measurements help explain why the samples may be difficult to distinguish. However, the model uses all 30 features, so the four measurements alone cannot explain the errors completely.

The missed malignant sample is particularly important because false negatives can carry significant consequences in medical classification.

### Numerical Gradient Checking

I compared the analytically derived gradients with gradients estimated using finite differences.

The maximum relative difference was approximately:

`3.7 × 10⁻¹⁰`

This small difference provides evidence that the analytical gradient implementation is correct to numerical precision.

## Project Structure

```text
breast-cancer-classification/
├── breast_cancer_model.py
├── plots.py
├── figures/
│   ├── fig1_korelasyon.png
│   ├── fig2_kayip_egrisi.png
│   ├── fig3_karisiklik_matrisi.png
│   ├── fig4_bolme_degiskenligi.png
│   ├── fig5_esik_analizi.png
│   └── fig6_gradyan_kontrolu.png
├── .gitignore
└── README.md
```

`breast_cancer_model.py` contains the model implementation, experiments, and main workflow.

`plots.py` generates six visualizations and saves them to the `figures/` directory.

## How to Run

Clone the repository:

```bash
git clone https://github.com/senturkilayda/breast-cancer-classification.git
cd breast-cancer-classification
```

Install the dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

Run the project:

```bash
python breast_cancer_model.py
```

The script prints the experiment results and saves the generated figures in the `figures/` directory.

To enable exploratory data analysis plots, set `RUN_EDA = True` near the beginning of the script.

## Visualizations

| File | Description |
|---|---|
| `fig1_korelasyon.png` | Correlation matrix of 10 mean features |
| `fig2_kayip_egrisi.png` | Training and test loss across 100 epochs |
| `fig3_karisiklik_matrisi.png` | Confusion matrix |
| `fig4_bolme_degiskenligi.png` | Accuracy and malignant recall across 30 splits |
| `fig5_esik_analizi.png` | False positives and false negatives at different decision thresholds |
| `fig6_gradyan_kontrolu.png` | Analytical vs. numerical gradient comparison |

## Limitations

- The dataset is relatively small and comes from a single source. Performance on data from other hospitals or populations remains unknown.
- There is no separate validation set. Choosing hyperparameters or thresholds based on test results could lead to overly optimistic performance estimates.
- The threshold analysis illustrates the trade-off between false positives and false negatives. It was not used to select a new decision threshold.
- This is an educational machine learning project and is not intended for clinical diagnosis or decision-making.

## Next Steps

1. Implement L2 regularization manually and examine its effect on model performance.
2. Build a small neural network with a hidden layer and derive backpropagation manually.
3. Reimplement the network in PyTorch and compare autograd gradients with manually calculated gradients.

## References

1. Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993). Nuclear feature extraction for breast tumor diagnosis. *Proceedings of SPIE*, 1905, 861–870. https://doi.org/10.1117/12.148698
2. Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). *Breast Cancer Wisconsin (Diagnostic)* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B
3. Varoquaux, G. (2018). Cross-validation failure: Small sample sizes lead to large error bars. *NeuroImage*, 180(A), 68–77. https://doi.org/10.1016/j.neuroimage.2017.06.061
4. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. http://www.deeplearningbook.org/
5. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
6. Harris, C. R., et al. (2020). Array programming with NumPy. *Nature*, 585, 357–362. https://doi.org/10.1038/s41586-020-2649-2
