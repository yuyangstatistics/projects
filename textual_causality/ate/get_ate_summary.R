library(dplyr)

get.res.print <- function(ate_res, save_path) {
  ate_res_print <- as.data.frame(ate_res) %>% 
    group_by(Setting, alpha, beta, gamma) %>%
    summarise(y.mean.sd = sd(y.mean),
              y.mean = mean(y.mean), 
              Truth = mean(Truth), 
              Unadjusted.sd = sd(Unadjusted), 
              Unadjusted = mean(Unadjusted), 
              PSReg.sd = sd(PSReg), 
              PSReg = mean(PSReg), 
              PSS.sd = sd(PSS),
              PSS = mean(PSS), 
              IPW.sd = sd(IPW), 
              IPW = mean(IPW), 
              IPW2.sd = sd(IPW2),
              IPW2 = mean(IPW2)) %>%
    mutate(y.mean = paste(round(y.mean, 4), "(", 
                          round(y.mean.sd, 5), ")", sep = ""), 
           Truth = round(Truth, 4), 
           Unadjusted = paste(round(Unadjusted, 4), "(", 
                              round(Unadjusted.sd, 5), ")", sep = ""), 
           PSReg = paste(round(PSReg, 4), "(", 
                         round(PSReg.sd, 5), ")", sep = ""), 
           PSS = paste(round(PSS, 4), "(", 
                       round(PSS.sd, 5), ")", sep = ""), 
           IPW = paste(round(IPW, 4), "(", 
                       round(IPW.sd, 5), ")", sep = ""), 
           IPW2 = paste(round(IPW2, 4), "(", 
                        round(IPW2.sd, 5), ")", sep = "") 
    ) %>% 
    select(alpha, beta, gamma, y.mean, Truth, Unadjusted, PSReg, PSS, IPW, IPW2)
  
  ate_res_print <- as.data.frame(ate_res_print)
  rownames(ate_res_print) <- paste("Setting", ate_res_print$Setting)
  ate_res_print <- ate_res_print[, 2:ncol(ate_res_print)]
  write.csv(ate_res_print, file = save_path)
}


ate_res <- read.csv("../results/ate_results.csv")
ate_res <- ate_res[, 2:ncol(ate_res)]
get.res.print(ate_res, "../results/ate_results_print.csv")

ate_res <- read.csv("../results/ate_results2.csv")
ate_res <- ate_res[, 2:ncol(ate_res)]
get.res.print(ate_res, "../results/ate_results_print2.csv")