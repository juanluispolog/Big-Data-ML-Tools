from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
import time
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.types import StructType, StructField, FloatType
from pyspark.ml.feature import VectorAssembler
from sklearn.metrics import confusion_matrix

spark = SparkSession.builder.appName('SparkProject').getOrCreate()

filename = "SUSY"
#filename = "Alumns"

schema_SUSY = StructType([
    StructField("idx", FloatType(), True),
    StructField("col0", FloatType(), True),
    StructField("col1", FloatType(), True),
    StructField("col2", FloatType(), True),
    StructField("col3", FloatType(), True),
    StructField("col4", FloatType(), True),
    StructField("col5", FloatType(), True),
    StructField("col6", FloatType(), True),
    StructField("col7", FloatType(), True),
    StructField("col8", FloatType(), True),
    StructField("col9", FloatType(), True),
    StructField("col10", FloatType(), True),
    StructField("col11", FloatType(), True),
    StructField("col12", FloatType(), True),
    StructField("col13", FloatType(), True),
    StructField("col14", FloatType(), True),
    StructField("col15", FloatType(), True),
    StructField("col16", FloatType(), True),
    StructField("col17", FloatType(), True),
    StructField("col18", FloatType(), True)])

schema_Alumns = StructType([
    StructField("idx", FloatType(), True),
    StructField("col0", FloatType(), True),
    StructField("col1", FloatType(), True),
    StructField("col2", FloatType(), True),
    StructField("col3", FloatType(), True)])

start_read = time.time()

if filename == "SUSY":
    schema = schema_SUSY
else:
    schema = schema_Alumns

df_train = spark.read.csv('/data/' + filename + "train.csv", schema=schema)
df_train = df_train.drop(df_train.columns[0])

df_test = spark.read.csv('/data/' + filename + "test.csv", schema=schema)
df_test = df_test.drop(df_test.columns[0])
end_read = time.time()

df_columns = df_train.columns

start_train = time.time()
assembler = VectorAssembler(inputCols=df_columns, outputCol="features")
df_train = assembler.transform(df_train)
df_test = assembler.transform(df_test)

lr = LogisticRegression(labelCol="col0", featuresCol="features", maxIter=2, regParam=0.3, elasticNetParam=0.5)

model = lr.fit(df_train)

predict_train = model.transform(df_train)
predict_test = model.transform(df_test)
print(predict_test.select("col0", "prediction").show(10))
end_train = time.time()

start_test = time.time()
evaluator = MulticlassClassificationEvaluator(labelCol="col0", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predict_test)

end_test = time.time()
print(filename)
print("Test Error = %g" % (1.0 - accuracy))
print("Accuracy = %g\n" % accuracy)

y_true = predict_test.select("col0")
y_true = y_true.toPandas()

y_pred = predict_test.select("prediction")
y_pred = y_pred.toPandas()

cnf_matrix = confusion_matrix(y_true, y_pred)
print(cnf_matrix)

print("------------------------------")
print("Read time: " + str(end_read-start_read))
print("Train time: " + str(end_train-start_train))
print("Test time: " + str(end_test-start_test))
print("Total time: " + str(end_test-start_read))
