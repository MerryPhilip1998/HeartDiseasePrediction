# Package Installation
list.of.packages <- c("caret","tidyverse", "yardstick","plotly", "forcats",
                      "kernlab","e1071", "mlbench", "mice", "ggplot2", "GGally") 

new.package <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.package)){
  install.packages(new.package)}

# Install the libraries

library(caret)              #For ML
library(tidyverse)          #For data manipulation
library(yardstick)          #For ROC curve
library(plotly)             #For plots
library(forcats)            #For ordering the labels
library(e1071)              #For SVM
library(mice)               #For missing data
library(ggplot2)
library(GGally)
library(reshape2)
options(scipen=999)         #For exponential outputs

# Loading the data set
data_df <- read_csv("C:/Users/merry/OneDrive/Documents/HDS/Assignment/Classification/heart_disease_modified.csv")

#Examining the data
glimpse(data_df)
str(data_df)

#Counting number of unique values in a column
apply(data_df, 2, function(x) length(unique(x)))

duplicates <- data_df %>% 
  group_by(Patient_ID) %>% 
  filter(n()>1)

"As there are only 1 value in 'pace_maker', this column will be removed to avoid
overfitting along with 'patient_id' and first column (...1)"

data_df <- data_df %>% 
  select(-c("pace_maker", "Patient_ID", "...1"))

#Data Transformation
"Converting drug and fam_hist column to numeric"
data_df$fam_hist <- ifelse(data_df$fam_hist == "yes", 1, 0)


data_df$drug <- ifelse(data_df$drug == "Aspirin", 1, 
                       ifelse(data_df$drug == "Clopidogrel", 2,
                              ifelse(data_df$drug == "Both", 3, 4)))

#Relabeling the class with 1 as 'disease' and 0 as'no_disease'
data_df$class <- factor(data_df$class, levels = c(1,0), labels = c("disease", "no_disease"))
data_df$class <- fct_relevel(data_df$class, "disease","no_disease")

str(data_df)

#Missing Values
sum(is.na(data_df))

#Explanatory Data Analysis (EDA)
table(data_df$class)

#Hist for labeled class
ggplot(data_df, aes(y = class)) + 
  geom_bar(aes(fill = class)) +
  xlab("Heart Disease") +
  ylab("Count") +
  theme(legend.position = "top")


#list of binary and non binary columns
bin <- c("sex", "fbs", "exang", "smoker","fam_hist")

non_binary <- c("age", "cp", "trestbps", "chol", "restecg", "thalach",
                "oldpeak", "perfusion", "slope", "ca", "thal", "traponin",  "drug")

#For proportion of disease and no disease of BINARY VARIABLES
prop = aggregate(x = data_df[,bin],
                 by = list(data_df$class),
                 FUN = sum)

prop[1, 2:6] <- prop[1, 2:6] / sum(data_df$class == 'disease')
prop[2, 2:6] <- prop[2, 2:6] / sum(data_df$class == 'no_disease')
prop

"This depicts that Male (1) has higher chances of heart_disease than Female.
Also, fasting blood sugar > 120, exercice induced angina has higher chances of heart_disease.
However, Smoking and Family histroy has no much difference."

#Probability density for Non_BINARY VARIABLES
featurePlot(x = data_df[, non_binary],      # from Caret
            y = data_df$class,
            plot = 'density',               # but its a good practice
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")), 
            adjust = 1.5, 
            pch = "|", 
            layout = c(3, 2), 
            auto.key = list(columns = 2))


#Data Imputation
#Simulating missing values

dfmis <- data_df %>%
  select(-class)#Do not need to to impute labels

#converting 0 in chol, thalach and trestbps 0 to NA
dfmis$chol[dfmis$chol == 0] <- NA
dfmis$thalach[dfmis$thalach == 0] <- NA
dfmis$trestbps[dfmis$trestbps == 0] <- NA

df.imputed <- mice(dfmis)
df.complete <- complete(df.imputed)

summary(df.complete)

#Dataset with Imputed data
data_df <- cbind(df.complete, "class" = data_df$class)


featurePlot(x = data_df[, c("chol", "thalach", "trestbps")],      # from Caret
            y = data_df$class,              # as it featureplot can detect by itself.
            plot = 'density',               # but its a good practice
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")), 
            adjust = 1.5, 
            pch = "|", 
            layout = c(3, 2), 
            auto.key = list(columns = 2))


"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

#Data Splitting
#Function for splitting
splitData <- function(someDataset,trainPercent) {
  
  bound <- nrow(someDataset)*trainPercent   # set training threshold
  someDataset <- someDataset[sample(nrow(someDataset)),]    # shuffle
  someDatasetTrain <- someDataset[1:bound,]
  someDatasetTest <- someDataset[(bound+1):nrow(someDataset),]
  
  obj <- list("train" = someDatasetTrain, "test" = someDatasetTest)
  return(obj)
}

heartdata_split <- splitData(data_df,0.70)

#Correlation
#Correlation test
### Correlation Matrix
cormat <- round(x = cor(heartdata_split$train[,non_binary]), digits = 2)


### Let's melt the corralation matrix
melted_cormat <- melt(cormat)

ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill = value)) +
  geom_tile()


#Data Preprocessing
"As the data is in various range, we will preprocess the data to range 0 to 1"
"Preprocessing done for testing dataset"
#Normalizing the training data to 0 to 1(Range)

preprocessing <- preProcess(x = heartdata_split$train,
                            method = c('range'))

preprocessing

#Predicting preprocessed training set
tr_preprocessed <- predict(preprocessing, heartdata_split$train)
test_preprocessed <- predict(preprocessing, heartdata_split$test)

test_prepro_noclass <- test_preprocessed %>% select(-"class")


"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
#Running data using default parameters

# Train naiveBayes
nb <- naiveBayes(class ~., data = tr_preprocessed)
#Testing the model
nb.predict <- predict(nb, test_prepro_noclass)

cm <- table(predicted = nb.predict, observed = test_preprocessed$class)

cm_bal <- confusionMatrix(nb.predict, test_preprocessed$class, positive = 'disease')

nb.accuracy <- round(100*(cm[1,1] + cm[2,2]) / nrow(heartdata_split$test),2)

paste0("Accuracy is: ",nb.accuracy,"%")

"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
#Running data using default parameters
nb <- train(class ~., 
            data = tr_preprocessed, 
            method = "nb",
            trControl = trainControl(classProbs =  TRUE, savePred =T))


#Testing the model
nb.predict <- predict(nb, test_prepro_noclass)

cm <- table(predicted = nb.predict, observed = heartdata_split$test$class)

cm_bal <- confusionMatrix(nb.predict, test_preprocessed$class, positive = 'disease')

nb.accuracy <- round(100*(cm[1,1] + cm[2,2]) / nrow(heartdata_split$test),2)

paste0("Accuracy is: ",nb.accuracy,"%")


#ROCNB
rocnb <- nb$pred %>% roc_auc(obs,disease) %>%
  select(.estimate) %>%
  round(.,2) %>% 
  cbind(nb$pred, "AUC" = paste0("svm AUC = ", .))



plotnbFit <- rocnb %>% roc_curve(obs,disease) %>%
  autoplot + theme(text = element_text(size=20)) + aes(label=round(.threshold,6)) 

plotnbFit %>% ggplotly(print.auc=TRUE, print.thres='best', grid=TRUE)

#AUC
nb_auc <- nb$pred %>% roc_auc(obs,disease)
nb_auc
"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

svmFitLinear <- train(class ~., 
                      data = tr_preprocessed, 
                      method = "svmLinear",
                      trControl = trainControl(classProbs =  TRUE, savePred =T))

svmFitLinear

#Predict the values with

#Removing class from test dataset
test_prepro_noclass <- test_preprocessed %>% select(-class) 

svm.predict <- predict(svmFitLinear, test_prepro_noclass) #Predicting for test

svm_cm <- table(predicted = svm.predict, observed = heartdata_split$test$class)

cm_bal <- confusionMatrix(svm.predict, test_preprocessed$class, positive = 'disease')

cm_bal
svm.accuracy <- round(100*(svm_cm[1,1] + cm[2,2]) / nrow(heartdata_split$test),2)

paste0("Accuracy is: ",svm.accuracy,"%")

"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
#Train SVM

svmFit <- train(class ~., 
                data = tr_preprocessed, 
                method = "svmRadial")

svmFit

#Training the entire tarining set with obtained C and sigma value
params <- data.frame(C = svmFit$bestTune$C,sigma=svmFit$bestTune$sigma)

svmFit <- train(class ~., 
                data = tr_preprocessed, 
                method = "svmRadial",
                tuneGrid=params,
                trControl = trainControl(classProbs =  TRUE, savePred =T))

svmFit
#Predict the values with
svm.predict <- predict(svmFit, test_prepro_noclass) #Predicting for test

svm_cm <- table(predicted = svm.predict, observed = heartdata_split$test$class)

cm_bal <- confusionMatrix(svm.predict, test_preprocessed$class, positive = 'disease')

cm_bal
svm.accuracy <- round(100*(svm_cm[1,1] + cm[2,2]) / nrow(heartdata_split$test),2)

paste0("Accuracy is: ",svm.accuracy,"%")

svm.accuracy
nb.accuracy

"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
##ROC for SVM Linear Model

#ROC
rocsvm <- svmFitLinear$pred %>% roc_auc(obs,disease) %>%
  select(.estimate) %>%
  round(.,2) %>% 
  cbind(svmFit$pred, "AUC" = paste0("svm AUC = ", .))

plotsvmFit <- rocsvm %>% roc_curve(obs,disease) %>%
  autoplot + theme(text = element_text(size=20)) + aes(label=round(.threshold,6)) 

plotsvmFit %>% ggplotly(print.auc=TRUE, print.thres='best', grid=TRUE)

#AUC
svmLinear_auc <- svmFitLinear$pred %>% roc_auc(obs,disease)
svmLinear_auc
"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
##ROC for SVM Radial Model

#ROC
rocsvm <- svmFit$pred %>% roc_auc(obs,disease) %>%
  select(.estimate) %>%
  round(.,2) %>% 
  cbind(svmFit$pred, "AUC" = paste0("svm AUC = ", .))



plotsvmFit <- rocsvm %>% roc_curve(obs,disease) %>%
  autoplot + theme(text = element_text(size=20)) + aes(label=round(.threshold,6)) 

plotsvmFit %>% ggplotly(print.auc=TRUE, print.thres='best', grid=TRUE)

#AUC
svmRadial_auc <- svmFit$pred %>% roc_auc(obs,disease)
svmRadial_auc

"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
#Optimizing SVM Model

control <- trainControl(method="repeatedcv", number=10, 
                        savePredictions = TRUE, 
                        classProbs = TRUE)


svmFitGridL <- train(class ~., data = tr_preprocessed,
                     method = "svmLinear",
                     trControl = control,
                     tuneLength = 10)
svmFitGridL$pred


#Radial 
svmFitGridR <- train(class ~., data = tr_preprocessed,
                     method = "svmRadial",
                     trControl = control,
                     tuneLength = 10)

svmFitGridR

"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

n <- 0
svm.accuracy <- numeric(15)
#Cross Fold Validation
for (train_idx in createMultiFolds(tr_preprocessed$class, k = 10, times = 3)){
  n <- n + 1
  
  #Getting Indexed data
  df_tr <- tr_preprocessed[train_idx, c(bin, non_binary, "class")] 
  
  #svm
  svm <- train(class ~., data = df_tr,
               method = "svmRadial",
               trControl = trainControl(method = "repeatedcv",number=10, 
                                        savePredictions = TRUE, 
                                        classProbs = TRUE,
                                        search = "random"),
               number = length(df_tr),
               tuneLength = 10)
  
  svm.predict_test <- predict(svm, test_preprocessed)
  
  svm.cm <- confusionMatrix(svm.predict_test, test_preprocessed$class, positive = 'disease')
  
}

#Variable Importance
importance <- varImp(svm, scale=FALSE)
plot(importance)

#ROC
resultsvmFit <- svm$pred %>% roc_auc(obs,disease) %>%
  select(.estimate) %>%
  round(.,2) %>% 
  cbind(svm$pred, "AUC" = paste0("svm AUC = ", .))



plotsvmFit <- resultsvmFit %>% roc_curve(obs,disease) %>%
  autoplot + theme(text = element_text(size=20)) + aes(label=round(.threshold,6)) 

plotsvmFit %>% ggplotly(print.auc=TRUE, print.thres='best', grid=TRUE)



