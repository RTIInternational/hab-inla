# Author: Natalie Reynolds
# Date created:10/11/2024

# Purpose:
# This code contains an example model for workflow development
# It will read in data, fit a simple INLA model, and export the model as an R data object
# There will be a secondary script to read in the saved model and issue predictions on brand new data

# Set up workspace ----
# Clear environment
rm(list = ls())

# Read packages
library(tidyverse)
library(rstudioapi)
library(fs)
library(janitor)
library(inlabru)
library(INLA)
library(ROCR)
library(caret)
library(sf)

# Set working directory - might need to change this for running on Linux server
script_path <- rstudioapi::getActiveDocumentContext()$path
setwd(dirname(script_path))

# Import files ----
# I'm intentionally limiting the model formulation so we don't have to have all datasets included in this example code. However, make sure you add the met data here later on if the model formulation is edited to require it.

# File paths and names

lakes_fp <- 'data/HABS_Prediction_Polygons.shp'

cyan_data_fp <- 'data/stage/cyan_lake_stats_7day.csv'

static_data_fp <- "data/HABS_Model_Static_Data.csv"

# Import files

fl_lz <- read_sf(lakes_fp) %>% clean_names() 

cyan_data <- read_csv(cyan_data_fp) %>% clean_names() %>% dplyr::select(-x1)

static_data <- read_csv(static_data_fp) %>% clean_names() 

# Fit mesh ----
# Separate out lakes and zones
fl_lz %>% filter(type == 'Zone') %>%
  pull(lake_id) %>% unique() -> lakes_with_zones # List of lakes that have zones

fl_lz %>% filter(type == "Lake") -> fl_lakes # Lakes only

# Removes whole lakes for the nine that have zones
fl_lz %>% filter(!id %in% as.numeric(str_c(as.character(lakes_with_zones),'000'))) -> fl_zones 

# Pull centroids for creating mesh. 
loc <- tibble(
  lon = st_coordinates(st_centroid(fl_zones))[,1],
  lat = st_coordinates(st_centroid(fl_zones))[,2]
)

max_edge <- diff(range(loc[,1]))/3

# Fit mesh
mesh <- inla.mesh.2d(
  loc = loc, 
  boundary = fl_lakes,
  max.edge = max_edge*c(1,2),
  offset = max_edge*c(0.75,1),
  cutoff = 5000,
  min.angle = 30
)

# Scale data and prep cyan input data ----
# Prep static_data - scale all the values
static_data %>% 
  # Remove cols for precip/temp anomaly 
  dplyr::select(colnames(.)[1:18]) %>%
  # Scale predictor vars
  mutate(
    wtr_pixl_cnt = scale(wtr_pixl_cnt)[,1],
    wtr_area_km2 = scale(wtr_area_km2)[,1],
    total_area_km2 = scale(total_area_km2)[,1],
    depth_hyd_m = scale(depth_hyd_m)[,1],
    max_fetch_m = scale(max_fetch_m)[,1],
    geo_mean_color = scale(geo_mean_color)[,1],
    geo_mean_alk = scale(geo_mean_alk)[,1],
    avg_precip_annual = scale(avg_precip_annual)[,1],
    avg_temp_annual = scale(avg_temp_annual)[,1]
  ) -> static_data_prepped

# Set aside dates for prediction and verification
verification_dates <- c("20220423","20230114","20230722","20230909") %>% ymd()

# Prediction time step: This would not be coded in for an operational model because we would be issuing predictions for dates that haven't occurred yet. However, for the sake of not having to edit my existing code too much, I'm keeping it in. The code will still yield a trained model that can be applied to future dates.
max_date <- max(cyan_data$date) 

# Prep cyan data
cyan_data %>% 
  # Remove data prior to max_date - In a deployed model, this shouldn't remove any rows
  filter(date <= max_date) %>% 
  
  # Comment out the following once Kris corrects the data
  rename(id = lake_id) %>% 
  
  # Calculate bloom/no bloom
  ## EDIT HERE TO CHANGE BLOOM DEFINITION ----
mutate(bloom = ifelse(perc_ge_thresh>=0.1,1,0)) %>%
  
  # Set aside last time step for prediction and the verification dates for
  mutate(subset = ifelse(date == max_date,'prediction',ifelse(date %in% verification_dates,'verification',NA))) %>%
  
  # Pre-process bloom data; start by removing missing bloom data for rows not in prediction set
  filter(!is.na(subset) | (!is.na(bloom) & is.na(subset))) %>% 
  
  # Group by feature to make sure all sampling is even across lakes (or zones where applicable)
  group_by(id) %>%
  
  # Assign subsets for training and validation
  mutate(
    # Create a row number column for if/else statement
    row = ifelse(is.na(subset), row_number(),NA),
    # Use the rows to randomly sample the data
    subset = ifelse(date == max_date,'prediction', # Keep the prediction data
                    ifelse(date %in% verification_dates, 'verification', # Set aside verification dates for model evaluation
                           ifelse(row %in% sample(
                             x = max(row, na.rm = T), 
                             size = floor(0.7 * max(row, na.rm = T))
                           ),
                           'training',# Assign 70% to training
                           'validation'))), # Remaining data is reserved as validation data
    subset = factor(subset)) %>% 
  
  # Ungroup data
  ungroup() %>% 
  
  # Order data
  arrange(id,date) %>% 
  
  # Go ahead and prep input col (NA's for val and pred datasets)
  mutate(bloom_input = ifelse(subset != "training", NA, bloom)) %>%
  
  # Join with other datasets
  left_join(static_data_prepped) %>%
  
  # Remove unused columns
  dplyr::select(id,date,perc_ge_thresh,bloom,bloom_input,subset,colnames(static_data_prepped)) -> cyan_data_prepped

# Attach spatial data - make this as a separate obj because tibbles are sometimes easier to work with 
fl_lz %>% 
  st_centroid() %>% 
  dplyr::select(id,name,type,geometry) %>%
  full_join(cyan_data_prepped) %>%
  arrange(id,date) %>% as_Spatial() -> cyan_data_point

# Set up latent model components ----
# Spatial:
spde <- inla.spde2.matern(mesh)
# Temporal:
hyprior <- list(theta1 = list(prior="pc.prec", param=c(0.5, 0.05)),
                theta2 = list(prior="pc.cor1", param=c(0.1, 0.9)))

# Set up subsets to later fit model cutoff value ----
# Pull subsets - should be the same across all models
training_subset <- which(cyan_data_prepped$subset == "training")
validation_subset <- which(cyan_data_prepped$subset == "validation")
verification_subset <- which(cyan_data_prepped$subset == "verification")
prediction_subset <- which(cyan_data_prepped$subset == "prediction")

# Pull observations
training_observations <- cyan_data_prepped$bloom[training_subset]
validation_observations <- cyan_data_prepped$bloom[validation_subset]
verification_observations <- cyan_data_prepped$bloom[verification_subset]
prediction_observations <- cyan_data_prepped$bloom[prediction_subset]

# Set up model assessment functions ----
# Confusion matrix
generate_cm <- function(responses, observations,cutoff) {
  tibble(
    response = factor(responses >= cutoff[3],
                      levels = c(T,F), ordered = T),
    observed = factor(observations, levels = c(1,0),
                      labels = c(T,F), ordered = T)
  ) %>% table() %>% return()
}

# Performance stats
generate_stats <- function(responses, observations,cutoff) {
  cm <- generate_cm(responses = responses, observations = observations,cutoff = cutoff)
  
  roc_instance <- prediction(responses,observations)
  
  auc <- performance(roc_instance, measure = 'auc')
  
  stats <- confusionMatrix(cm)
  
  tibble(
    AUC = auc@y.values[[1]],
    Sensitivity = stats$byClass[1],
    Specificity = stats$byClass[2],
    Accuracy = stats$overall[1],
    Precision = cm[1,1]/(cm[1,1] + cm[1,2]),
    Prevalence = (cm[1,1] + cm[2,1])/sum(cm),
    `False Omission Rate` = cm[2,1]/(cm[2,1] + cm[2,2]),
    `F1 Score` = (2*cm[1,1])/(2*cm[1,1] + cm[1,2] + cm[2,1]),
    Kappa = stats$overall[2],
    `Brier Score` = mean((responses - observations)^2)
  ) %>% t() %>% return()
}

# Model cutoff
optimize_cutoff <- function(performance, value) {
  cut_index <- mapply(FUN = function(x, y, p){
    d <- (x - 0)^2 + (y - 1)^2
    index <- which(d == min(d))
    c(sensitivity = y[[index]], # tpr
      specificity = 1-x[[index]], #tnr
      cutoff = p[[index]])},
    # The '@' operator is used to access data in S4 class objects (e.g. the performance instances)
    performance@x.values, # tpr
    performance@y.values, # tnr
    value@cutoffs
  )
}

# Wrapper function that implements the above three functions (need this to optimize cutoff)
calculate_performance_stats <- function(model){ # ex: model = "fit_1"
  
  fit_i <- get(model)
  subscript <- str_extract(model, "(?<=_).*") 
  formula_i <- get(str_c("formula_",subscript))
  
  cat(str_c("\nGenerating summary stats for ",model))
  # cat(str_c("\nModel call: ", str_c(as.character(formula_1)[2]," ~ ", as.character(formula_1)[3])))
  
  # Pull predictions if generated
  if(class(try(get(str_c("predictions_",subscript)), silent = T)) == "try-error") {
    # cat(str_c("\nPredictions have not been generated for ", subscript,
    #           ";\nSummary stats will only be generated for training and validation subsets"))
  } else {
    predictions_i <- get(str_c("predictions_",subscript))
  }
  
  # Pull responses and save to environment
  responses_i <- fit_i$summary.fitted.values$mean[1:nrow(cyan_data)]
  assign(str_c("responses_",subscript), responses_i, envir = parent.frame())
  
  training_responses_i <- responses_i[training_subset]
  assign(str_c("training_responses_",subscript), training_responses_i, envir = parent.frame())
  validation_responses_i <- responses_i[validation_subset]
  assign(str_c("validation_responses_",subscript), validation_responses_i, envir = parent.frame())
  verification_responses_i <- responses_i[verification_subset]
  assign(str_c("verification_responses_",subscript), verification_responses_i, envir = parent.frame())
  
  # Calculate cutoff value
  validation_roc_instance_i <- prediction(validation_responses_i,validation_observations)
  assign(str_c("validation_roc_instance_",subscript),validation_roc_instance_i, envir = parent.frame())
  validation_roc_performance_instance_i <- performance(validation_roc_instance_i, measure = 'tpr', x.measure = 'fpr')
  assign(str_c("validation_roc_performance_instance_",subscript),validation_roc_performance_instance_i, envir = parent.frame())
  
  cutoff_i <- optimize_cutoff(validation_roc_performance_instance_i,
                              validation_roc_instance_i) # Cutoff value is cutoff[3]
  assign(str_c("cutoff_",subscript),cutoff_i, envir = parent.frame())
  
  # Print cutoff value
  # cat(str_c("\nCutoff value: ",cutoff_i[3]))
  
  # Generate performance stats
  ### TRAINING RESULTS
  # cat("\nTraining results:\n")
  train_cm_i <- generate_cm(responses = training_responses_i,observations = training_observations,cutoff_i)
  # print(train_cm_i); 
  assign(str_c("train_cm_",subscript),train_cm_i, envir = parent.frame())
  train_stats_i <- generate_stats(responses = training_responses_i,
                                  observations = training_observations,cutoff_i)
  # print(train_stats_i); 
  assign(str_c("train_stats_",subscript),train_stats_i, envir = parent.frame())
  
  ### VALIDATION RESULTS
  # cat("\nValidation results:\n")
  val_cm_i <- generate_cm(responses = validation_responses_i,observations = validation_observations,cutoff_i)
  # print(val_cm_i); 
  assign(str_c("val_cm_",subscript),val_cm_i, envir = parent.frame())
  val_stats_i <- generate_stats(responses = validation_responses_i,
                                observations = validation_observations,cutoff_i)
  # print(val_stats_i); 
  assign(str_c("val_stats_",subscript),val_stats_i, envir = parent.frame())
  
  ### VERIFICATION RESULTS
  # cat("\nVerification results:\n")
  ver_cm_i <- generate_cm(responses = verification_responses_i,observations = verification_observations,cutoff_i)
  # print(ver_cm_i); 
  assign(str_c("ver_cm_",subscript),ver_cm_i, envir = parent.frame())
  ver_stats_i <- generate_stats(responses = verification_responses_i[which(!is.na(verification_observations))],
                                observations = verification_observations[which(!is.na(verification_observations))],cutoff_i)
  # print(ver_stats_i); 
  assign(str_c("ver_stats_",subscript),ver_stats_i, envir = parent.frame())
  
  ## PREDICTION RESULTS
  if(try(exists("predictions_i"), silent = T) == F) {
    cat(str_c("\nFinished processing results for fit_",subscript,"\n"))
  } else {
    # cat("\nPrediction results:\n")
    
    prediction_responses_i <- predictions_i$mean
    
    # Need to remove missing data from prediction observations
    pred_auc_susbset <- which(!is.na(prediction_observations))
    
    pred_cm_i <- generate_cm(responses = prediction_responses_i[pred_auc_susbset],observations = prediction_observations[pred_auc_susbset],cutoff_i)
    # print(pred_cm_i); 
    assign(str_c("pred_cm_",subscript),pred_cm_i, envir = parent.frame())
    pred_stats_i <- generate_stats(responses = prediction_responses_i[pred_auc_susbset],
                                   observations = prediction_observations[pred_auc_susbset],cutoff_i)
    # print(pred_stats_i); 
    assign(str_c("pred_stats_",subscript),pred_stats_i, envir = parent.frame())
    cat(str_c("\nFinished processing results for fit_",subscript,"\n"))
  }
  
  # Combine all stats for function output
  colnames(train_stats_i) <- 'training'
  colnames(val_stats_i) <- 'validation'
  colnames(ver_stats_i) <- 'verification'
  if(try(exists("predictions_i"), silent = T) == T){
    colnames(pred_stats_i) <- 'prediction'
    return(list(
      str_c("Model call for ", model,": ", str_c(as.character(formula_i)[2]," ~ ", as.character(formula_i)[3])),
      cbind(train_stats_i,val_stats_i,ver_stats_i,pred_stats_i)
    ))
  } else {
    return(list(
      str_c("Model call for ", model,": ", str_c(as.character(formula_i)[2]," ~ ", as.character(formula_i)[3])),
      cbind(train_stats_i,val_stats_i,ver_stats_i)
    ))
  }
  
}

# Fit a simple model ----
# Replace formula_ex whatever model formulation we actually use, THIS IS JUST TO BE USED AN EXAMPLE FOR THE WORKFLOW
formula_ex <- bloom_input ~ 
  total_area_km2 +
  depth_hyd_m +
  geo_mean_color +
  geo_mean_alk +
  avg_precip_annual +
  avg_temp_annual +
  s(geometry,model = spde) + t(yday(date), model = "ar1",hyper = hyprior, constr = T) + Intercept(1)

fit_ex <- bru(formula_ex,
              data = cyan_data_point, 
              family = "binomial",
              control.family = list(link = 'logit'),
              options = bru_options(control.predictor = list(compute = T,
                                                             link = 1),
                                    control.inla = list(int.strategy = "eb"),
                                    control.compute = list(dic = T,cpo = T,config = T))
              )

# View model summary
summary(fit_ex)

# Generate cutoff value and view performance stats
calculate_performance_stats('fit_ex')

# Print cutoff value:
cutoff_ex[[3]]

# Save relevant R objects:
# Objects that won't be renamed:
save(loc, file = 'model_components.Rdata')

# Objects that will be renamed:
# Model
saveRDS(fit_ex,file = 'model_fit.RDS')
# Fit
saveRDS(cutoff_ex[[3]], file = 'model_cutoff.RDS')

# Prediction dataset - THIS IS SIMPLY SO I DON'T HAVE TO RECREATE A NEW DATASET IN THE NEXT SCRIPT
saveRDS(cyan_data_point[which(cyan_data_point$subset == 'prediction'),],file = 'prediction_dataset.RDS')
