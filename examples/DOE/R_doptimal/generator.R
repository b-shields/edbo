
##############################Optimal Experimental Design#########################  Amination

install.packages("AlgDesign")

library(AlgDesign)

cand  <-  gen.factorial(levels=c(3,22,3,4),
                        nVars= 4,
                        factors="all", varNames = c("aryl_halide",
                                                    "additive",
                                                    "base",
                                                    "ligand"))

set.seed(0)
des  <-  optFederov( ~.^2, 
                    data=cand, 
                    nTrials = 100,
                    approximate = FALSE)

path <- "C:/Users/Ben/Dropbox/_project-updates/bayesian-opt/_parallel_mauscript/revisions-1/doe/R_doptimal/amination_d-optimal.csv"
write.csv(des$design, path)

##################################################################################  Suzuki

cand  <-  gen.factorial(levels=c(4,3,7,11,4),
                        nVars= 5,
                        factors="all", varNames = c("electrophile",
                                                    "nucleophile",
                                                    "base",
                                                    "ligand",
                                                    "solvent"))

set.seed(0)
des  <-  optFederov( ~ ., 
                     data=cand, 
                     nTrials = 45,
                     approximate = FALSE)

path <- "C:/Users/Ben/Dropbox/_project-updates/bayesian-opt/_parallel_mauscript/revisions-1/doe/R_doptimal/suzuki_d-optimal-5.csv"
write.csv(des$design, path)

##################################################################################
