

# import libraries
require('mltools')
require('caTools')
require('e1071')
require('randomForest')
require('foreach')
require('doParallel')
require('abind')


#' Get binned probabilities.
#'
#' @param y.true A numeric vector. The true response.
#' @param y.prob A numeric vector. The predicted probabilites.
#' @param nbins  An integer. The number of bins.
#'
#' @return A list with two elements. The first element is the binned actual fractions, and
#'   the second element is the binned predicted probabilites.
#' @export
#'
#' @examples
#' y.true <- c(0, 0, 1, 1, 0, 1, 1, 0, 0, 0)
#' y.prob <- c(0.45454545, 0.36363636, 0.63636364, 0.18181818, 0.45454545, 0.09090909,
#' 0.27272727, 0.81818182, 0.63636364, 0.63636364)
#' binProbs(y.true, y.prob)
#'
binProbs <- function(y.true, y.prob, nbins=5){

  stopifnot(is.numeric(y.true) & is.numeric(y.prob))
  stopifnot(length(y.true) == length(y.prob))

  # consider only uniform case temporarily
  bins <- seq(0, 1+1e-8, length.out = nbins+1)
  bin.ids <- as.factor(as.numeric(bin_data(y.prob, bins = bins, binType = "explicit")))
  output <- list(prob.true = sapply(split(y.true, bin.ids), mean),
                 prob.pred = sapply(split(y.prob, bin.ids), mean))
  return (output)
}


#' Standard Brier Score
#'
#' @param y.true A numeric vector. The true response.
#' @param y.prob A numeric vector. The predicted probabilites.
#'
#' @return A numeric scalar. The standard Brier score.
#' @export
#'
#' @examples
#' y.true <- c(0, 0, 1, 1, 0, 1, 1, 0, 0, 0)
#' y.prob <- c(0.45454545, 0.36363636, 0.63636364, 0.18181818, 0.45454545, 0.09090909,
#' 0.27272727, 0.81818182, 0.63636364, 0.63636364)
#' brier(y.true, y.prob)
#'
#' @details
#' Brier score measures the fit of probability estimates to the observed data.
#'   It is defined as the mean squared difference between the observed labels the
#'   estimated probability. A smaller value means a better calibration.
#'   \deqn{BS = \frac{\sum_{i=1}^N (y_i - \hat{P}(y_i | x_i))^2}{N}}
#'
brier <- function(y.true, y.prob) {

  stopifnot(is.numeric(y.true) & is.numeric(y.prob))
  stopifnot(length(y.true) == length(y.prob))

  return(mean((y.true - y.prob)**2))
}


#' Stratified Brier Score
#'
#' @param y.true A numeric vector. The true response.
#' @param y.prob A numeric vector. The predicted probabilites.
#'
#' @return A list with three elements. The 1st is the standard Brier score, The 2nd is the Brier score
#'   for the positive class, and the 3rd is the Brier score for the negative class.
#' @export
#'
#' @examples
#' y.true <- c(0, 0, 1, 1, 0, 1, 1, 0, 0, 0)
#' y.prob <- c(0.45454545, 0.36363636, 0.63636364, 0.18181818, 0.45454545, 0.09090909,
#' 0.27272727, 0.81818182, 0.63636364, 0.63636364)
#' stratifiedBrier(y.true, y.prob)
#'
#' @details
#' Stratified Brier Score evaluates the goodness of calibration under the imbalanced scenario.
#'   \deqn{BS^+ = \frac{\sum_{y_i=\text{pos_label}} (y_i - \hat{P}(y_i | x_i))^2}{N_{pos}}}
#'   \deqn{BS^- = \frac{\sum_{y_i=\text{neg_label}} (y_i - \hat{P}(y_i | x_i))^2}{N_{pos}}}
#'
#' @references
#' Wallace, B.C., Dahabreh, I.J. Improving class probability estimates for imbalanced data.
#'   Knowl Inf Syst 41, 33–52 (2014). https://doi.org/10.1007/s10115-013-0670-6
#'
stratifiedBrier <- function(y.true, y.prob) {

  stopifnot(is.numeric(y.true) & is.numeric(y.prob))
  stopifnot(length(y.true) == length(y.prob))

  bs <- mean((y.true - y.prob)**2)
  bs.plus <- mean((y.true - y.prob)[y.true == 1]**2)
  bs.minus <- mean((y.true - y.prob)[y.true == 0]**2)

  return(list('BS' = bs, 'BS+' = bs.plus, 'BS-' = bs.minus))
}


#' Plot calibration diagram (reliability plot).
#'
#' @param y.true A numeric vector. The true response.
#' @param y.prob A numeric vector. The predicted probabilites.
#' @param mod.name A string. The name of the model you want to plot, used in the legend.
#' @param nbins  An integer. The number of bins.
#'
#' @return A plot with two subplots. The top one is the calibration curve plot, and the bottom
#'   one is the histogram of the predicted probabilities.
#' @export
#'
#' @examples
#' y.true <- c(0, 0, 1, 1, 0, 1, 1, 0, 0, 0)
#' y.prob <- c(0.45454545, 0.36363636, 0.63636364, 0.18181818, 0.45454545, 0.09090909,
#' 0.27272727, 0.81818182, 0.63636364, 0.63636364)
#' calibCurve(y.true, y.prob, 'Example')
#'
calibCurve <- function(y.true, y.prob, mod.name, nbins=5){

  stopifnot(is.numeric(y.true) & is.numeric(y.prob))
  stopifnot(length(y.true) == length(y.prob))

  calib.out <- binProbs(y.true, y.prob, nbins = nbins)
  brier.score <- brier(y.true, y.prob)

  layout(matrix(c(1, 1, 1, 2), 4, 1, byrow = TRUE))

  plot(c(0, 1), c(0, 1), lty = 2, col = 'black', type = 'l',
       main = 'Calibration Curve (Reliability Diagram)', xlab = "Mean predicted value",
       ylab = "Fraction of positives", xlim = c(0, 1), ylim = c(0, 1))

  lines(calib.out$prob.pred, calib.out$prob.true, type = 'o', lty = 1, col = 'blue')

  legend.name <- paste(mod.name, " (", round(brier.score, 4), ")", sep = "")
  legend("bottomright",
         legend = c(legend.name, "Perfectly calibrated"),
         lty = c(1, 2), col = c("blue", "red"), cex = 0.5)

  hist(y.prob, breaks = nbins + 1, main = "", xlab = "Mean predicted value")

}

#' Plot comparison among calibrations.
#'
#' @description
#' Do comparisons among different models or different calibration methods, in terms of the
#'   calibrated probabilities.
#'
#' @param y.true A numeric vector. The true response.
#' @param y.prob A numeric vector. The predicted probabilites.
#' @param mod.name A string. The name of the model you want to plot, used in the legend.
#' @param nbins  An integer. The number of bins.
#'
#' @return A plot with calibration curves given by different calibrations.
#' @export
#'
#' @examples
#' library('e1071')
#' data(imbalance)
#' set.seed(123)
#' split <- sample.split(imbalance$y, SplitRatio = 0.75)
#' train_set <- subset(imbalance, split == TRUE)
#' test_set <- subset(imbalance, split == FALSE)
#' X.test <- subset(test_set, select = -y)
#' y.test <- subset(test_set, select = y)[,1]
#' lr <- glm(y ~ ., data = train_set, family = "binomial")
#' prob.lr <- predict(lr, X.test, type = "response")
#' calibCurve(y.test, prob.lr, "Logistic")
#' nb <- naiveBayes(y ~ ., data = train_set)
#' prob.nb <- predict(nb, X.test, type = "raw")[, 2]
#' calibCurve(y.test, prob.nb, "Naive Bayes")
#' comparisonPlot(y.test, list(prob.lr, prob.nb), c("Logistic", "Naive Bayes"))
#'
#' @references
#' \href{https://scikit-learn.org/stable/auto_examples/calibration/plot_compare_calibration.html}{sklearn: Comparison of Calibration of Classifiers}
#'
comparisonPlot <- function(y.true, probs, mod.names, nbins=5){

  # probs should be a list, mod.names should be a vector
  n <- length(probs)
  brier.score <- unlist(lapply(probs, function(x) brier(y.true, x)))

  plot(c(0, 1), c(0, 1), lty = 2, lwd = 1, type = 'l', col = 1,
       main = "Comparison of Calibration Curve",
       xlab = "Mean predicted value", ylab = "Fraction of positives",
       xlim = c(0, 1), ylim = c(0, 1))

  for (i in 1:n){
    calib.out <- binProbs(y.true, probs[[i]], nbins = nbins)
    lines(calib.out$prob.pred, calib.out$prob.true, type = 'o', lty = 1, lwd=2, col = i+1)
  }

  legend.names <- mapply(function(name, score) {
    paste(name, " (", round(score, 4), ")", sep = "")},
    mod.names, brier.score)
  legend("bottomright", legend = c("Perfectly calibrated", legend.names),
         lty = c(2, rep(1, n)), lwd = c(1, rep(2, n)), col = 1:(n+1), cex = 0.5)

}


#' Dataset under-sampling
#'
#' @param pos_set A dataframe. The dataframe for the positive class.
#' @param neg_set A dataframe. The dataframe for the negative class.
#'
#' @return A dataframe. An under-sampled dataset, with equal number of positive class and
#'   negative class.
#' @export
#'
underSample <- function(pos_set, neg_set) {

  stopifnot(dim(pos_set)[2] == dim(neg_set)[2])

  sample_pos_set <- pos_set[sample(nrow(pos_set), nrow(pos_set), replace = TRUE),]

  sample_neg_set <- neg_set[sample(nrow(neg_set), nrow(pos_set), replace = TRUE),]

  sample_set <- rbind(sample_pos_set, sample_neg_set)

  return(sample_set)
}




#' Bagged undersampled calibration.
#'
#' @description Bagged undersampled calibration, can do both the weighted average and
#'   simple average on the bagged probabilities.
#'   Parallel computing is enabled to speed up the bagging procedure by specifying `ncluster`.
#'   Choose between weighted average of bagged probabilities or simple average using `ntimes`.
#'
#' @details The simple average and the weighted average of the bagged probabilities are defined
#'   as below.
#'
#'   Simple average:
#'   \deqn{\hat{P}(y_i | x_i) = \frac{1}{k}\sum_{j=1}^k \hat{P}_j(y_i | f_{ij})}
#'
#'   Weighted average:
#'   \deqn{\hat{P}(y_i | x_i) = \frac{1}{z}\sum_{j=1}^k \frac{1}{\text{Var}(\hat{P}_j(y_i | f_{ij}))}  \hat{P}_j(y_i | f_{ij}),}
#'   where \deqn{z = \sum_{j=1}^k \frac{1}{\text{Var}(\hat{P}_j(y_i | f_{ij}))}.}
#'
#' @param trainset A dataframe. The training dataset.
#' @param newX An array. The feature matrix of the new test data.
#' @param response_name A string. The name of the response column in the training dataset.
#' @param model A string. The model to calibrate. Options: `'svm'`, `'lr'`, `'nb'`, `'rf'`.
#' @param formula A formula. The formula of the model.
#' @param pos_label An integer. 0 or 1. The label for the positive class.
#' @param nbags An integer. How many sample set are used for bagging. Note that a large value
#'   will not lead to overfitting and will reduce the variance more, with the only cost being
#'   heavy computation load.
#' @param ntimes An integer. The number of times to run the model within each sample set.
#'   When `ntimes=1`, the output is the simple average of `nbags` sets of probabilities,
#'   and when `ntimes > 1`, the output is the weighted average, with the weight being the empirical
#'   variance of `ntimes` predicted probabilities within each sample set.
#' @param ncluster An integer. The number of clusters to use in the parallel implementaion.
#' @param ... Arguments with variable lengths. The extra arguments for specifying the model.
#'
#' @return A numeric vector. The calibrated probabilities by weighted bagged undersampled method.
#' @export
#'
#' @examples
#' data(imbalance)
#' set.seed(123)
#' split <- sample.split(imbalance$y, SplitRatio = 0.75)
#' train_set <- subset(imbalance, split == TRUE)
#' test_set <- subset(imbalance, split == FALSE)
#' X.test <- subset(test_set, select = -y)
#' y.test <- subset(test_set, select = y)[,1]
#' # standard calibration
#' svc <- svm(formula = as.factor(y) ~ ., data = train_set, type = 'C-classification',
#'   kernel = 'linear', probability = TRUE)
#' pred <- predict(svc, X.test, probability = TRUE)
#' dec.svc <- attr(pred, 'decision.values')
#' prob.svc <- as.data.frame(attr(pred, "probabilities"))$`1`
#' stratifiedBrier(y.test, prob.svc)
#' # calibration using bagged undersampling method (simple average)
#' bag.prob.svm <- bagCalibrate(train_set, X.test, 'y', model='svm', type = 'C-classification',
#'   kernel = 'linear', nbags = 30, ntimes = 1, ncluster = 4)
#' stratifiedBrier(y.test, bag.prob.svm)
#' # calibration using weighted bagged undersampling method (weighted average)
#' weighted.bag.prob.svm <- bagCalibrate(train_set, X.test, 'y', model='svm', type = 'C-classification',
#'   kernel = 'linear', nbags = 30, ntimes = 20, ncluster = 4)
#' stratifiedBrier(y.test, weighted.bag.prob.svm)
#'
#' # comparison plot
#' comparisonPlot(y.test, list(prob.svc, bag.prob.svm, weighted.bag.prob.svm),
#'   c("SVM", "bagged-under SVM", "Weighted-bagged-under SVM"), nbins = 8)
#'
#' @references
#' Wallace, B.C., Dahabreh, I.J. Improving class probability estimates for imbalanced data.
#'   Knowl Inf Syst 41, 33–52 (2014). https://doi.org/10.1007/s10115-013-0670-6
#'
bagCalibrate <- function(trainset, newX, response_name, model, formula=as.factor(y) ~ .,
                                                 pos_label=1, nbags = 25, ntimes = 1, ncluster=4, ...) {

  stopifnot(response_name %in% names(trainset))

  # change response name to 'y'
  if (response_name != 'y') {
    colnames(trainset)[names(trainset) == response_name] <- 'y'
  }

  # obtain positive set and negative set
  pos_set <- subset(trainset, y == pos_label)
  neg_set <- subset(trainset, y != pos_label)

  # define the combine function used in foreach
  if (ntimes > 1) {
    cfun <- function(...) {abind(..., along=3)}
  } else {
    cfun <- function(...) {abind(..., along=2)}
  }

  #setup parallel back end to use 4 processors
  cl<-makeCluster(4)
  registerDoParallel(cl)

  wprobs <-foreach(j=1:nbags, .combine=cfun, .multicombine = TRUE, .packages = c('foreach'),
                  .export = c('underSample', 'cfun')) %dopar% {

                    # obtain sample set
                    sample_set <- underSample(pos_set, neg_set)

                    # for each sample set, run ntimes to obtain the empirical variance, and hence the weighted probs
                    probs <- foreach(i=1:ntimes, .combine = cbind, .packages = c('e1071', 'randomForest')) %do% {

                                      if (model == 'svm') {

                                        clf <- svm(formula = formula, data = sample_set, ..., probability = TRUE)
                                        pred <- predict(clf, newX, probability = TRUE)
                                        prob_df <- as.data.frame(attr(pred, "probabilities"))   # predicted probability
                                        colnames(prob_df)[names(prob_df) == pos_label] <- 'positive'
                                        prob <- prob_df$positive

                                      } else if (model == 'lr') {

                                        clf <- glm(formula = formula, data = sample_set, family = "binomial", ...)
                                        prob <- predict(clf, newX, type = "response")

                                      } else if (model == 'nb') {

                                        clf <- naiveBayes(formula = formula, data = sample_set, ...)
                                        prob_df <- as.data.frame(predict(clf, newX, type = "raw"))
                                        colnames(prob_df)[names(prob_df) == pos_label] <- 'positive'
                                        prob <- prob_df$positive

                                      } else if (model == 'rf') {

                                        clf <- randomForest(formula = formula, data = sample_set, ...)
                                        prob_df <- as.data.frame(predict(clf, newX, type = "prob"))
                                        colnames(prob_df)[names(prob_df) == pos_label] <- 'positive'
                                        prob <- prob_df$positive

                                      }

                                      prob

                                    }

                    if (ntimes > 1) {
                      # obtain median and variance of the probabilities, use 1 / variance as the weight
                      # return the n*2 array to the outer foreach
                      cbind(apply(probs, 1, median), 1/apply(probs, 1, var))
                    } else {
                      probs
                    }

                  }

  stopCluster(cl)

  if (ntimes > 1) {

    # wprobs is three-dim array of size: n*2*nbags
    P <- wprobs[, 1, ]   # probabilities
    W <- wprobs[, 2, ]   # weights
    # return weighted average of probabilities
    return (rowMeans(P * (W / rowMeans(W)) ))

  } else {

    # return simple average of probabilities
    return (rowMeans(wprobs))

  }

}


