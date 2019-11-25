setwd("/Users/juanluispolog/Downloads")

begin <- Sys.time()
data <- read.csv("Alumns.csv", header = T)
end <- Sys.time()
time.import <- end - begin

str(data)
head(data)

data[, 2:4] <- scale(data[, 2:4])

library(caret)
set.seed(1122)
train.rows <- createDataPartition(data[,1], p = 0.8, list = F)
train <- data[train.rows,]
test <- data[- train.rows,]

begin <- Sys.time()
model <- glm(admit ~ ., data = train, family = binomial)
end <- Sys.time()
time.model <- end-begin

begin <- Sys.time()
prob <- predict(model, test, type = "response")
end <- Sys.time()
time.pred <- end-begin

pred <- ifelse(prob > 0.5, "1", "0")
confusionMatrix(as.factor(pred), as.factor(test[,1]))

time.import
time.model
time.pred
