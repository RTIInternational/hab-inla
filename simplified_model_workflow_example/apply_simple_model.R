# Author: Natalie Reynolds
# Date created:10/11/2024

# Purpose:
# This code reads in the model from train_simple_model.R
# It will also read in data and issue predictions for a new date

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

# File paths
# Model components:
model_comp_fp <- 'model_components.Rdata'

# Model and cutoff:
fit_fp <- 'model_fit.RDS'
cutoff_fp <- 'model_cutoff.RDS'

# DATASET NOTE: I'm not sure what this workflow would look like in practice, whether the data will be prepped in the prediction script or in a separate script. For the sake of simplicity, I'm reading in the prediction subset that was set aside in train_simple_model.R.
prepped_data_fp <- 'prediction_dataset.RDS' 

# Read in files
load(model_comp_fp)
fit <- readRDS(fit_fp)
cutoff <- readRDS(cutoff_fp)
prediction_data <- readRDS(prepped_data_fp)

# Prep data ----
# In a real prediction dataset, we won't have Bloom, Bloom_input, or perc_ge_thresh - these are included as relics for how I selected my code
# You MUST have your data formatted as a SpatialPointsDataFrame and you must have every column in your prediction call listed in the dataset, EVEN IF THOSE COLUMNS ARE EMPTY.
prediction_data <- prediction_data[,c(1:4,8:ncol(prediction_data))]

predictions <- predict(fit, 
                         prediction_data,
                       # You have to specify all of your model components here:
                       ~ exp(total_area_km2 +
                               depth_hyd_m +
                               geo_mean_color +
                               geo_mean_alk +
                               avg_precip_annual +
                               avg_temp_annual +
                               s + t + Intercept)/(1 + exp(total_area_km2 +
                                                             depth_hyd_m +
                                                             geo_mean_color +
                                                             geo_mean_alk +
                                                             avg_precip_annual +
                                                             avg_temp_annual +
                                                             s + t + Intercept)), # converts log-odds to prob,
                       ## As log-odds:
                       # ~total_area_km2 +
                       # depth_hyd_m +
                       # geo_mean_color +
                       # geo_mean_alk +
                       # avg_precip_annual +
                       # avg_temp_annual +
                       # s + t + Intercept, # results as log-odds
                       n.samples = 500,
                       seed = 1234)

# Export data ----
predictions %>%
  as_tibble() %>%
  select(id,name,wbid,type,date, mean) %>%
  rename(bloom_prob = mean) %>%
  # Encodes predicted bloom as 1 and no predicted bloom as 0
  mutate(bloom_pred = ifelse(bloom_prob >= cutoff, 1, 0)) -> predictions_output

print(predictions_output)

write_csv(predictions_output, "example_prediction.csv")

