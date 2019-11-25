setwd("/Users/juanluispolog/Downloads")

library(caret)
begin.total <- Sys.time()
begin <- Sys.time()
data <- read.csv("SUSY.csv", header = F)
end <- Sys.time()
time.import <- end-begin

str(data)
head(data)

train <- data[1:4000000,]
test <- data[4000001:5000000,]

begin <- Sys.time()
model <- glm(V1 ~ ., data = train, family = binomial)
end <- Sys.time()
time.model <- end-begin

begin <- Sys.time()
prob <- predict(model, test, type = "response")
end <- Sys.time()
time.pred <- end-begin
end.total <- Sys.time()

time.total <- end.total - begin.total

pred <- ifelse(prob > 0.5, "1", "0")
confusionMatrix(as.factor(pred), as.factor(test[,1]))

time.import
time.model
time.pred
time.total
