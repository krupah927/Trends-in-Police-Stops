
// $example on$
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FilterFunction;
import org.apache.spark.api.java.function.ForeachFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import static org.apache.spark.sql.functions.*;
import org.apache.spark.sql.*;


import static org.apache.zookeeper.server.ZooKeeperServer.ok;
// $example off$

public class MLtree {
    public static void main(String[] args) {
        JavaSparkContext sc = new JavaSparkContext(new SparkConf().setAppName("policingDT").setMaster("local[*]"));
        SQLContext sqlContext = new SQLContext(sc);
        SparkSession spark = SparkSession
                .builder()
                .appName("MLtreek")
                .getOrCreate();

        // $example on$
        // Load the data stored in LIBSVM format as a DataFrame.
        Dataset<Row> data = spark
                .read()
                .format("csv")
                .load("CO_cleaned.csv");
       Dataset<Row> filterData= data.drop("_c1").drop("_c2").drop("_c3").drop("_c4").drop("_c5").drop("_c6").drop("_c7")
               .drop("_c8").drop("_c10").drop("_c12").drop("_c14").drop("_c15").drop("_c16").drop("_c17")
               .drop("_c18").drop("_c19").drop("_c20").drop("_c22").drop("_c23").drop("_c24").drop("_c25")
               .withColumnRenamed("_c9","gender").withColumnRenamed("_c13","race")
               .withColumnRenamed("_c11","age").withColumnRenamed("_c21","labelvalue");





        filterData.createOrReplaceTempView("data");
       // Dataset<Row> ok=newdata.filter(newdata.col("age").isNotNull(),newdata.col("labelValue").isNotNull());
        Dataset<Row> newdata = filterData.filter((FilterFunction<Row>) row -> !row.anyNull());

        // Index labels, adding metadata to the label column.
        // Fit on whole dataset to include all labels in index.
        StringIndexerModel labelIndexer = new StringIndexer()
                .setInputCol("labelvalue")
                .setOutputCol("labelCol")
                .fit(newdata);
       // Dataset<Row> labelednewdata = labelIndexer.transform(newdata);

        //convert race
        StringIndexerModel index1 = new StringIndexer()
                .setInputCol("race")
                .setOutputCol("raceIndex")
                .fit(newdata);

        Dataset<Row> indexed1 = index1.transform(newdata);

    //convert gender
        StringIndexerModel index2 = new StringIndexer()
                .setInputCol("gender")
                .setOutputCol("genIndex")
                .fit(indexed1);

        Dataset<Row> indexed2 = index2.transform(indexed1);

        //convert age
        StringIndexerModel index3 = new StringIndexer()
                .setInputCol("age")
                .setOutputCol("ageIndex")
                .fit(indexed2);

        Dataset<Row> indexed3 = index3.transform(indexed2);
        indexed3.show();

        //convert to vector
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"raceIndex"})
                .setOutputCol("features");

        Dataset<Row> outputD = assembler.transform(indexed3);


        VectorIndexerModel featureIndexer = new VectorIndexer()
                .setInputCol("features")
                .setOutputCol("indexedFeatures")
                .setMaxCategories(6) // features with > 4 distinct values are treated as continuous.
                .fit(outputD);

        // Split the data into training and test sets (30% held out for testing).
        Dataset<Row>[] splits = outputD.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

        // Train a DecisionTree model.
        DecisionTreeClassifier dt = new DecisionTreeClassifier()
                .setLabelCol("labelCol")
                .setFeaturesCol("indexedFeatures");

        // Convert indexed labels back to original labels.
        IndexToString labelConverter = new IndexToString()
                .setInputCol("prediction")
                .setOutputCol("predictedLabel")
                .setLabels(labelIndexer.labels());

        // Chain indexers and tree in a Pipeline.
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{labelIndexer, featureIndexer, dt, labelConverter});

        // Train model. This also runs the indexers.
        PipelineModel model = pipeline.fit(trainingData);

        // Make predictions.
        Dataset<Row> predictions = model.transform(testData);

        // Select example rows to display.
        predictions.select("gender","race","predictedLabel", "labelvalue", "features").show(15);

        // Select (prediction, true label) and compute test error.
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("labelCol")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Test Error = " + (1.0 - accuracy));

        DecisionTreeClassificationModel treeModel =
                (DecisionTreeClassificationModel) (model.stages()[2]);
        System.out.println("Learned classification tree model:\n" + treeModel.toDebugString());
        // $example off$

        spark.stop();
    }
}