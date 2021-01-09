# R package: imbCalib

This is a class project for STAT 8054 (Statistical Methods IV).

This package intends to calibrate probabilities for imbalanced data. The method is to do bagging over undersampled datasets, and hence to mitigate the bias of probability calibration induced by the imbalance of the dataset. Refer to [Wallace et.al 2014](https://doi.org/10.1007/s10115-013-0670-6) for more detail.

## Datasets
`imbalance`: An imbalanced dataset with 8 covariates and 1 binary response. There are about 5% samples in the positive class.

## Functions
- `binProbs`: generate binned probabilities.
- `brier`: calculate standard Brier score.
- `stratifiedBrier`: calculate stratified Brier score.
- `undersample`: generate an undersampled dataset.
- `bagCalibrate`: perform bagged undersampled calibration
  - Support logistic regression, naive Bayes, random forest, and svm currently.
- `calibCurve`: plot calibration curve.
- `comparisonPlot`: plot calibration curves by several calibrated probabilities.

## Vignettes

Check the link: [Vignette: Getting started with imbCalib](https://yuyangyy.com/assets/courses/files/vignette-imbcalib.html).

Or, install the package and then
```r
browseVignettes("imbCalib")
```

## References:

Wallace, B.C., Dahabreh, I.J. Improving class probability estimates for imbalanced data. Knowl Inf Syst 41, 33–52 (2014). https://doi.org/10.1007/s10115-013-0670-6

## Helpful Resources to understand probability calibration
[The Basics of Classifier Evaluation: Part 2](http://www.svds.com/classifiers2/)
[How to Calibrate Probabilities for Imbalanced Classification](https://machinelearningmastery.com/probability-calibration-for-imbalanced-classification/)
[Calibration of probabilities for tree-based models](https://gdmarmerola.github.io/probability-calibration/)