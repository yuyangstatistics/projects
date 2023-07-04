library(dplyr)
library(reshape2)
library(ggplot2)

gen.plot <- function(ate_res, save_path) {
  ate_res_plot <- as.data.frame(ate_res) %>% 
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
              IPW2 = mean(IPW2))
  
  ate_res_plot <- melt(ate_res, 
                       id.vars = c("Setting", "rep", "alpha", "beta", "gamma", 
                                   "y.mean"), 
                       variable.name = "Method")
  
  ate_res_plot <- ate_res_plot %>% 
    group_by(Setting, alpha, beta, gamma, Method) %>% 
    summarise(val.avg = mean(value, na.rm = TRUE), 
              val.se = sd(value, na.rm = TRUE))
  
  gp <- ggplot(ate_res_plot, 
               aes_string(x = "beta", y = "val.avg", color = "Method")) + 
    xlab(expression(beta)) + ylab("ATE") + 
    ggtitle("Average Treatment Effect Estimation") + 
    scale_x_discrete(limits = factor(1:10)) + 
    geom_line(lwd = 1.5) + geom_point(pch = 19) +
    scale_color_manual(values = c("#000000", "#f76161", "#2990ff", "#712a95",
                                  "#27e838", "#ffc864", "#D55E00", "#CC79A7")) + 
    theme_bw() + theme(legend.position = "bottom", 
                       plot.title = element_text(hjust = 0.5, size = 25), 
                       legend.text = element_text(size = 15), 
                       axis.title = element_text(size = 20), 
                       axis.text = element_text(size = 20)) + 
    geom_errorbar(aes(ymin = val.avg - qnorm(0.975) * val.se, 
                      ymax = val.avg + qnorm(0.975) * val.se), width = 0.02, 
                  lwd = 1.5)
  
  ggsave(filename = save_path, plot = gp, width = 30, height = 25,
         units = "cm")
}


ate_res <- read.csv("../results/ate_results.csv")
ate_res <- ate_res[, 2:ncol(ate_res)]
gen.plot(ate_res, "../results/ate_results.pdf")

ate_res <- read.csv("../results/ate_results2.csv")
ate_res <- ate_res[, 2:ncol(ate_res)]
gen.plot(ate_res, "../results/ate_results2.pdf")




