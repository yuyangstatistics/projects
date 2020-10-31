#' Synthesized imbalanced data
#'
#' Data with 8 covariates and 1 binary response. The ratio of positive vs. negative class
#'   is 5:95. The positive label is 1.
#'
#' @docType data
#'
#' @usage data(imbalance)
#'
#' @keywords datasets
#'
#' @references \href{https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html}{sklearn.datasets.make_classification documentation}
#'
#' @source \href{https://raw.githubusercontent.com/yuyang-yy/materials/master/code/synthesize.py}{Python code to generate the data}
#'
#' @examples
#' data(imbalance)
#' set.seed(123)
#' split <- sample.split(imbalance$y, SplitRatio = 0.75)
#' train_set <- subset(imbalance, split == TRUE)
#' test_set <- subset(imbalance, split == FALSE)
#' X.test <- subset(test_set, select = -y)
#' y.test <- subset(test_set, select = y)[,1]
#' bag.prob.svm <- bagCalibrate(train_set, X.test, 'y', model='svm', type = 'C-classification', kernel = 'linear')
#' stratifiedBrier(y.test, bag.prob.svm)
"imbalance"

