# simulated data for test
y.true <- c(0, 0, 1, 1, 0, 1, 1, 0, 0, 0)
y.prob <- c(0.45454545, 0.36363636, 0.63636364, 0.18181818, 0.45454545, 0.09090909,
            0.27272727, 0.81818182, 0.63636364, 0.63636364)

# load imbalance data
data(imbalance)
set.seed(123)
split <- sample.split(imbalance$y, SplitRatio = 0.75)
train_set <- subset(imbalance, split == TRUE)
test_set <- subset(imbalance, split == FALSE)
X.test <- subset(test_set, select = -y)
y.test <- subset(test_set, select = y)[,1]


test_that("binProbs works", {

  expect_equal(as.numeric(binProbs(y.true, y.prob)[[1]][1]), 1)
  expect_equal(as.numeric(binProbs(y.true, y.prob)[[2]][1]), 0.13636363499999998306)

})

test_that("brier works", {

  expect_equal(brier(y.true, y.prob), 0.41818181890909089660)

})

test_that("stratifiedBrier works", {

  expect_equal(stratifiedBrier(y.true, y.prob)[[1]], 0.41818181890909089660)
  expect_equal(stratifiedBrier(y.true, y.prob)[[2]], 0.53925619983471062557)
  expect_equal(stratifiedBrier(y.true, y.prob)[[3]], 0.33746556495867768843)

})

test_that("bagCalibrate's output has the same length as y.test", {

  expect_equal(length(bagCalibrate(train_set, X.test, 'y', model='svm',
                                   type = 'C-classification', kernel = 'linear')), length(y.test))

})
