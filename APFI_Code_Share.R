# META ----
# Title:  Default setup for fainjj
# First Created: Fri May  7 10:12:17 2021
#
# NOTES ----
#'
#

# Setup ----
if (!require(rgeos)) { install.packages('rgeos') }; require(rgeos)
if (!require(rgdal)) { install.packages('rgdal') }; require(rgdal)
if (!require(raster)) { install.packages('raster') }; require(raster)
if (!require(tidyverse)) { install.packages('tidyverse') }; require(tidyverse)
if (!require(here)) { install.packages('here') }; require(here)
if (!require(sf)) { install.packages('sf') }; require(sf)
if (!require(spdep)) { install.packages('spdep') }; require(spdep)
if (!require(spatstat)) { install.packages('spatstat') }; require(spatstat)
if (!require(magrittr)) { install.packages('magrittr') }; require(magrittr)
if (!require(xgboost)) { install.packages('xgboost') }; require(xgboost)
if (!require(caret)) { install.packages('caret') }; require(caret)
if (!require(PNWColors)) { install.packages('PNWColors') }; require(PNWColors)
install.packages('cli'); library(cli)
install.packages("devtools") ; library(devtools)
devtools::install_github("AppliedDataSciencePartners/xgboostExplainer"); require(xgboostExplainer)
#


#  ----
if (!require(tidylog)) { install.packages('tidylog') }

library(tidylog)
# unloadNamespace(tidylog)
#


# Utility Functions ----
keep_input_vars <- . %>% select(Pers, nDays, nNeigh, FRP, LAT, TYPE)

scale01 <- function(x){
  (x - min(x)) / (max(x) - min(x))
}

lc_optimizer <- function(df) {
  NULL
}

add_explicit_LATLON <- function(df) {
  df %>%
    mutate(LAT= unlist(map(df$geometry,1)),
           LON = unlist(map(df$geometry,2)))
}
#


# Get fire data in study area ----
arctic_fires <- read_csv(here('Data', 'fires', 'arctic_2018', 'fire_archive_V1_155434.csv'))
#


# First load training data ----
omniread <- function(fpath) {
  is.csv <- grepl('csv', fpath)
  if(is.csv) {
    return(read_csv(fpath))
  }
  else {
    return(readxl::read_xlsx(fpath))
  }
}


# Labeled Data ----
lab_dat <- list.files(here('Data', 'training'),
                      recursive = T,
                      pattern = '(csv|xlsx)$', full.names = T) %>%
  map(omniread)

# 1-5, 7, & 10 all have a Pers column.
# 18-20 all have a filled Validation column.

has_pers <- lab_dat[c(1:5, 7, 10)] %>% {do.call(rbind, .)}
has_vali <- lab_dat[c(18:20)] %>% data.table::rbindlist(fill = T)

# These labels are indicating the same thing so let's combine them to make a
# smarter split.

has_vali <- mutate(has_vali, Pers = Validation)

combined_lab_dat <- bind_rows(keep_input_vars(has_vali), keep_input_vars(has_pers))
#


# Split Data ----
set.seed(12345)
starting_set <- sample(1:dim(combined_lab_dat)[1], 100)
base <- combined_lab_dat[starting_set, ]
pool <- combined_lab_dat[-starting_set, ]

# Use min dissimilarity for selecting sample
new_train_sample <- maxDissim(base, pool,
                              n = 700,
                              randomFrac = 0.1,
                              obj = minDiss)

# Split training out via ID
training_split_ids <- c(starting_set, new_train_sample)

# Separate training and testing
training_data <- combined_lab_dat[training_split_ids, ]
testing_data <- combined_lab_dat[-training_split_ids, ]
#


# Create xgb Data Matricies ----
training_mat <- xgb.DMatrix(data=as.matrix(training_data[ ,2:6]),
                            label=as.matrix(training_data$Pers))

testing_mat <- xgb.DMatrix(data = as.matrix(testing_data[ ,2:6]),
                           label = as.matrix(testing_data$Pers))
#


# Predict model with tuned parameters ----
tic() # Check time start
prediction_model <- xgboost(
  data = training_mat,
  eta = 0.01,
  max_depth = 100,
  nround=2000,
  subsample = 0.3,
  colsample_bytree = 0.2,
  eval_metric = "error",
  objective = "binary:logistic",
  booster = 'gbtree',
  early_stopping_rounds = 50
)
toc() # Compare runtime
#


# Predicted Outcomes ----
pred_on_test <- predict(prediction_model, testing_mat)


pred_on_test_bin <- vector('list', 4)
pred_on_test_bin[[1]] <- as.numeric(pred_on_test >= 0.65)
pred_on_test_bin[[2]] <- as.numeric(pred_on_test >= 0.70)
pred_on_test_bin[[3]] <- as.numeric(pred_on_test >= 0.80)
#


# Prediction Error ----
mean(pred_on_test_bin[[1]] != getinfo(testing_mat, 'label'))*100
mean(pred_on_test_bin[[2]] != getinfo(testing_mat, 'label'))*100
mean(pred_on_test_bin[[3]] != getinfo(testing_mat, 'label'))*100
#


# Model Metrics ----
explain_model <- xgboostExplainer::buildExplainer(prediction_model,
                                                  training_mat,
                                                  type = 'binary')


model_importance <- xgb.importance(model = prediction_model)

model_importance %>%
  xgb.ggplot.importance(n_clusters = 3) +
  theme_minimal() %+replace%
  theme(legend.position = 'none') +
  xlab(NULL)

model_importance %>%
  mutate(Variable = factor(Feature,
                           levels=c('nDays', 'TYPE',
                                    'nNeigh', 'LAT', 'FRP'),
                           labels=c('Unique Days', 'TYPE',
                                    'Unique Neighbors', 'Latitude', 'FRP')),
         label_pos = cumsum(Gain)-(Gain/2)) %>%
  ggplot() +
  geom_col(aes('', Gain, fill = fct_reorder(Feature, Gain)),
           position = 'stack',
           color='grey60') +
  geom_text(aes(x='', y=label_pos,
                label = Variable)) +
  scale_y_continuous(labels=scales::percent_format(accuracy = 0.01),
                     breaks = cumsum(
                       sort(model_importance$Gain, decreasing = T)
                     )
  ) +
  scale_fill_manual(NULL,
                    values=pnw_palette('Sailboat', 6, 'discrete')[-1],
                    guide='none') +
  labs(x=NULL) +
  coord_cartesian(expand=F) +
  theme_minimal()

ggsave(here('Grfx/cum_model_gain.png'), dpi = 500)
#


# Pare down columns and format names ----
f18 <- mutate(f18, LAT = st_coordinates(f18)[,2])

f18 <- select(f18, nDays, nNeigh, frp, LAT, type)

names(f18) <- c('nDays', 'nNeigh', 'FRP', 'LAT', 'TYPE', 'geometry')
#


# Predict and output for all fires in domain ----
f18 %>%
  st_drop_geometry() %>%
  select(nDays, nNeigh, FRP, LAT, TYPE) %>%
  as.matrix() %>%
  xgb.DMatrix() %>%
  {predict(prediction_model, .)} -> f18_predictions_xgboost

f18 <- f18 %>%
  select(nDays, nNeigh, FRP, LAT, TYPE) %>%
  mutate(Prob = f18_predictions_xgboost)

st_write(f18, here('Outs/fires_2018_predicted_xgboost.gpkg'))
#


#  ----
sd1_bounds <- c(low=mean(f18$Prob) - (sd(f18$Prob)/2),
                high=mean(f18$Prob) + (sd(f18$Prob)/2))

middle_sd1 <- f18 %>% filter(Prob >= sd1_bounds[1], Prob <= sd1_bounds[2])

nDay_weights <- middle_sd1$Prob*(1/sum(middle_sd1$Prob))

set.seed(42)

middle_sample_nday <- slice_sample(middle_sd1, n = 600,
                                   weight_by = nDay_weights,
                                   replace = F)
#


# Performance graphs ----
performance(
  prediction(
    pred_on_test,
    testing_data$Pers),
  'fpr', 'fnr') %>%
  plot(colorize=T)

performance(
  prediction(
    pred_on_test,
    testing_data$Pers),
  'prec', 'rec') %>%
  plot(colorize=T)

performance(
  prediction(
    pred_on_test,
    testing_data$Pers),
  'tpr', 'tnr') %>%
  plot(colorize=T)
#


# Testing probablilty cutoffs ----
f18 %>%
  filter(Prob >= 0.70) %>%
  st_write(here('Outs', 'fires_prob_70up.gpkg'))

f18 %>%
  filter(Prob >= 0.65) %>%
  st_write(here('Outs', 'fires_prob_65up.gpkg'))

raster(nrows=1000, ncols=2000,
       ymn=45, crs=4326) %>%
  `values<-`(1) %>%
  # writeRaster(here('Outs', 'temp_45_raster.tiff'), overwrite=TRUE)
  {.} -> r

# Check seasonality ----
f18_dates <- mutate(f18, acq_date = arctic_fires$acq_date)
f18_dates65 <- f18_dates %>% filter(Prob >= 0.65)
f18_dates65 <- f18_dates65 %>% mutate(month = lubridate::month(as.Date(acq_date)))
f18_dates65 <- f18_dates65 %>% mutate(month = lubridate::floor_date(as.Date(acq_date), unit="month"))

ggplot(f18_dates65 %>% st_drop_geometry()) +
  geom_bar(aes(x=month)) +
  scale_x_date(NULL, date_labels = "%b", breaks = "month") +
  scale_y_continuous(NULL, labels = scales::comma_format()) +
  theme_minimal() +
  ggtitle('Monthly Persistent Fire Observations', 'Predicted Probability 65% or Greater')
#


#  ----
table65 <- f18[f18$Prob >= 0.65,'TYPE'] %>% st_drop_geometry() %>% table()
(table65/nrow(f18[f18$Prob >= 0.65,]))*100


