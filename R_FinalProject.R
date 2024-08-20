# Install the package
#install.packages("glm2")

setwd("C:/Users/Laptop/Desktop/Belgium Campus iTversity/3rd Year/BIN371/BIN371_Milestone_3_Final Project")
data <- read.csv("BIN371_Milestone_3_Final_Project.csv")

#getwd()
#list.files()

# Load the package
library(glm2)

# Assuming 'data' is your prepared dataset and 'target' is your binary outcome variable
model <- glm(target ~ Annual.Salary + year_of_birth + household_size + yrs_residence, 
             data = data, 
             family = binomial(link = "logit"))

# View model summary
summary(model)

# Install the package
#install.packages("caret")

# Load required libraries
library(caret)

# Set seed for reproducibility
set.seed(123)

# Create index for training data (70% of the dataset)
train_index <- createDataPartition(data$target, p = 0.7, list = FALSE)

# Split the data
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Set up 10-fold cross-validation
ctrl <- trainControl(method = "cv", number = 10)

# Train the model with cross-validation
model_cv <- train(target ~ Annual.Salary + year_of_birth + household_size + yrs_residence,
                  data = train_data,
                  method = "glm",
                  family = binomial(link = "logit"),
                  trControl = ctrl)

# Build the final model on the entire training set
final_model <- glm(target ~ Annual.Salary + year_of_birth + household_size + yrs_residence,
                   data = train_data,
                   family = binomial(link = "logit"))

# View model summary
summary(final_model)

# Make predictions on the test set
predictions <- predict(final_model, newdata = test_data, type = "response")

# Convert probabilities to binary predictions
predicted_classes <- ifelse(predictions > 0.5, 1, 0)

# Create confusion matrix
conf_matrix <- table(Actual = test_data$target, Predicted = predicted_classes)

# Calculate accuracy
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)

# Calculate precision, recall, and F1 score
precision <- conf_matrix[2,2] / sum(conf_matrix[,2])
recall <- conf_matrix[2,2] / sum(conf_matrix[2,])
f1_score <- 2 * (precision * recall) / (precision + recall)

# Print results
print(conf_matrix)
print(paste("Accuracy:", accuracy))
print(paste("Precision:", precision))
print(paste("Recall:", recall))
print(paste("F1 Score:", f1_score))

#Visualizations
# Export ROC curve data
library(pROC)
roc_obj <- roc(test_data$target, predictions)
roc_data <- data.frame(FPR = 1 - roc_obj$specificities, TPR = roc_obj$sensitivities)
write.csv(roc_data, "roc_curve_data.csv", row.names = FALSE)

# Export feature importance data
importance_data <- data.frame(Feature = names(coef(final_model)[-1]), 
                              Importance = abs(coef(final_model)[-1]))
write.csv(importance_data, "feature_importance.csv", row.names = FALSE)

# Export prediction distribution data
pred_dist_data <- data.frame(Probability = predictions)
write.csv(pred_dist_data, "prediction_distribution.csv", row.names = FALSE)

# Export confusion matrix data
conf_matrix_data <- as.data.frame(conf_matrix)
write.csv(conf_matrix_data, "confusion_matrix.csv", row.names = FALSE)

# Export model performance metrics
metrics_data <- data.frame(Metric = c("Accuracy", "Precision", "Recall", "F1 Score"),
                           Value = c(accuracy, precision, recall, f1_score))
write.csv(metrics_data, "model_metrics.csv", row.names = FALSE)