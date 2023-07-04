library(ggplot2)
library(reshape2)
library(doParallel)
library(dplyr)

# this file works on the simulated data to estimate ATE
data <- read.csv("sim_data.csv")
head(data)

##=========  Tune the parameter settings ===========##
# gammas <- c(-2.45, -2.46, -2.47, -2.48)
# res <- foreach(gamma = gammas, .combine = "rbind") %dopar% {
#   c(gamma, ate.sim.comp(data, params = c(-1, 4, gamma)))
# }
# res

ate.sim.comp <- function(data, params, seed = 2021) {
  require(rms)
  set.seed(seed)
  data$y <- rbinom(nrow(data), 1, 
                   plogis(params[1] * data$Treat + params[2] * data$True_PS + 
                            params[3]))
  
  # True ATE
  ATE.true <- mean(plogis(params[1] + params[2] * data$True_PS + params[3])) - 
    mean(plogis(params[2] * data$True_PS + params[3]))
  
  # Unadjusted ATE
  ATE.unadj <- mean(data$y[data$Treat == 1]) - 
    mean(data$y[data$Treat == 0])
  
  # PS Regression Adjustment
  mod.ps <- glm(y ~ Treat * rcs(Est_PS, 5), data = data, 
                family = "binomial")
  data_trt <- data_ctr <- data
  data_trt$Treat <- 1
  data_ctr$Treat <- 0
  pred1.ps <- predict(mod.ps, newdata = data_trt, type = "response")
  pred0.ps <- predict(mod.ps, newdata = data_ctr, type = "response")
  ATE.PSReg <- mean(pred1.ps - pred0.ps)
  
  # PSS
  ps_quintile <- cut(data$Est_PS, 
                     breaks = c(0, quantile(data$Est_PS, p = 1:9 / 10 ), 1), 
                     labels = 1:10)
  table(ps_quintile, data$Treat)
  n <- nrow(data)
  nj <- table(ps_quintile)
  te_quintile <- tapply(data$y[data$Treat == 1], 
                        ps_quintile[data$Treat == 1], mean) - 
    tapply(data$y[data$Treat == 0], 
           ps_quintile[data$Treat == 0], mean)
  ATE.PSS <- sum(te_quintile * nj / n)
  
  # IPW
  w1 <- data$Treat / data$Est_PS
  w0 <- (1 - data$Treat) / (1 - data$Est_PS)
  ATE.IPW <- mean(data$y * w1) - mean(data$y * w0)
  
  # IPW2
  ATE.IPW2 <- weighted.mean(data$y, w1) - 
    weighted.mean(data$y, w0)
  
  
  res <- c(params, mean(data$y), 
           ATE.true, ATE.unadj, ATE.PSReg, ATE.PSS, ATE.IPW, ATE.IPW2)
  names(res) <- c("alpha", "beta", "gamma", "y.mean", "Truth", 
                  "Unadjusted", "PSReg", 
                  "PSS", "IPW", "IPW2")
  res
}


##=========================  Controlling mean of y ===========================##
# to keep mean of y nearly constant
get.param.setting <- function(param.type) {
  switch(param.type, 
         c(-1, 1, -1), 
         c(-1, 2, -1.6), 
         c(-1, 3, -2.1), 
         c(-1, 4, -2.7), 
         c(-1, 5, -3.2), 
         c(-1, 6, -3.75), 
         c(-1, 7, -4.3), 
         c(-1, 8, -4.85), 
         c(-1, 9, -5.4), 
         c(-1, 10, -6))
}
nrep <- 100
registerDoParallel(cores = 4)
ate_res <- c()
for (param.type in 1:10) {
  res <- foreach(rep = 1:nrep, .combine = "rbind") %dopar% {
    c(param.type, rep, 
      ate.sim.comp(data, params = get.param.setting(param.type), 
                   seed = 2021 + param.type + rep))
  }
  ate_res <- rbind(ate_res, res)
}
colnames(ate_res) <- c("Setting", "rep", "alpha", "beta", "gamma", "y.mean", 
                       "Truth", "Unadjusted", "PSReg", "PSS", "IPW", "IPW2")
write.csv(ate_res, file = "../results/ate_results.csv")


##=========================  Controlling ATE ===========================##
# to keep ATE nearly constant
get.param.setting2 <- function(param.type) {
  switch(param.type, 
         c(-1, 1, -1), 
         c(-1, 2, -1.5), 
         c(-1, 3, -2.0), 
         c(-1, 4, -2.47), 
         c(-1, 5, -2.95), 
         c(-1, 6, -3.4), 
         c(-1, 7, -3.8), 
         c(-1, 8, -4.15), 
         c(-1, 9, -4.45), 
         c(-1, 10, -4.5))
}

nrep <- 100
registerDoParallel(cores = 4)
ate_res <- c()
for (param.type in 1:10) {
  res <- foreach(rep = 1:nrep, .combine = "rbind") %dopar% {
    c(param.type, rep, 
      ate.sim.comp(data, params = get.param.setting2(param.type), 
                   seed = 2021 + param.type + rep))
  }
  ate_res <- rbind(ate_res, res)
}
colnames(ate_res) <- c("Setting", "rep", "alpha", "beta", "gamma", "y.mean", 
                       "Truth", "Unadjusted", "PSReg", "PSS", "IPW", "IPW2")
write.csv(ate_res, file = "../results/ate_results2.csv")

ate_res_print <- get.res.print(ate_res)
write.csv(ate_res_print, file = "../results/ate_results_print2.csv")

