#*
#*This script is a four-dimensional grid search written for
#*determining the best degrees of freedom for each of four natural
#*spline functions to be used in the GAM in the notebook general_additive_model.Rmd.
#*
#*During the grid-search on the degrees of freedom, we fit a 
#*spline to 90% of the training data and score this model against the 
#*other 10% of the training data. This scoring is performed 10 times
#*such that every subset of training data acts as a validation set for
#*the other 90% of the data. 
#*
#*These 10 validation scores are averaged and represent each the cross-validated
#*scores of each degree-of-freedom combination. This 10-fold cross-validation
#*results in an estimation of the model's test performance for every 
#*degree-of-freedom combination and allows us to combat over-fitting in the GAM, since
#*each validation subset serves as a proxy for the test set on which the model will
#*be finally evaluated. 
#*
#*The grid_search function prints a list of degrees of freedom for each parameter,
#* which can then be used in the GAM.
#*

grid_search <- function(input.train.file = "C:/Users/riley/Seismo_Direc/cleaned_training_df.csv",
                        input.test.file = "C:/Users/riley/Seismo_Direc/cleaned_testing_df.csv"){
  
  #*
  #*Loading Data & Libraries
  #*
  
  library(readr)
  library(Metrics)
  library(glmnet)
  library(boot)
  library(gam)
  library(splines)
  library(mltools)
  library(data.table)
  library(dplyr)
  
  train <- read_csv(input.train.file)
  test <- read_csv(input.test.file)
  
  #*
  #* Cross-Validation Grid Search
  #*
  
  #set a random seed
  set.seed(1)
  
  #make some vectors of indices to iterate over
  ml_dof <- 2:5
  fc_dof <- 2:7
  mc2_dof <- 1:5
  mws_dof <- 5:13
  
  #create some empty structures to hold our tuples and validation errors
  array_size <- length(ml_dof)*length(fc_dof)*length(mc2_dof)*length(mws_dof)
  cv.errors <- rep(NA, array_size)
  dof_combos <- vector(mode = "list", length = array_size)
  
  #starting index
  loop <- 1
  
  print("commencing cross-validation")
  
  #10-fold cross validation
  for(i in ml_dof){
    for(j in fc_dof){
      for(k in mc2_dof){
        for(l in mws_dof){
          
          #fit a gam
          temp.fit <- glm(Dep ~ ns(ML, df = i) + 
                            ns(fc, df = j) +
                            ns(Mc2, df = k) +
                            ns(Mws, df = l),
                            data = train)
          set.seed(1)
          #find the LOOCV MSE estimate
          cv.errors[loop] <- cv.glm(train, temp.fit, K=10)$delta[1]
          
          #capture the combination of degrees of freedom in a tuple
          dof_combos[[loop]] <- c(i,j,k,l)
          
          #increment the loop
          loop <- loop + 1
        }
      }
    }
    print("still going")
  }
  
  #compile these lists into an easily searchable data frame
  pseudo_dictionary <- data.frame(error = cv.errors,
                                  ML_dof = sapply(dof_combos, `[`, 1),
                                  fc_dof = sapply(dof_combos, `[`, 2),
                                  Mc2_dof = sapply(dof_combos, `[`, 3),
                                  Mws_dof = sapply(dof_combos, `[`, 4))
  
  #pick the row where the error is minimized
  winning_row <- pseudo_dictionary[pseudo_dictionary$error == min(pseudo_dictionary$error), ]
  print(winning_row)
  
}
  
  
  