% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/calibrate.R
\name{calibCurve}
\alias{calibCurve}
\title{Plot calibration diagram (reliability plot).}
\usage{
calibCurve(y.true, y.prob, mod.name, nbins = 5)
}
\arguments{
\item{y.true}{A numeric vector. The true response.}

\item{y.prob}{A numeric vector. The predicted probabilites.}

\item{mod.name}{A string. The name of the model you want to plot, used in the legend.}

\item{nbins}{An integer. The number of bins.}
}
\value{
A plot with two subplots. The top one is the calibration curve plot, and the bottom
one is the histogram of the predicted probabilities.
}
\description{
Plot calibration diagram (reliability plot).
}
\examples{
y.true <- c(0, 0, 1, 1, 0, 1, 1, 0, 0, 0)
y.prob <- c(0.45454545, 0.36363636, 0.63636364, 0.18181818, 0.45454545, 0.09090909,
0.27272727, 0.81818182, 0.63636364, 0.63636364)
calibCurve(y.true, y.prob, 'Example')

}
