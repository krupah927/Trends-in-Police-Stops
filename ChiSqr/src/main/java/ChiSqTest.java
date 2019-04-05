/****************************************************************************************************
 * Class ChiSqTest computes chisquare test
 * columns: age, race , gender - x axis
 * column search_conducted - yaxis
 ***************************************************************************************************/

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FilterFunction;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.stat.ChiSquareTest;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.SparkSession;


public class ChiSqTest {
    public static void main(String[] args) {

        JavaSparkContext sc = new JavaSparkContext(new SparkConf()
                .setAppName("Trafic trends Decision tree")
                .setMaster("local[*]"));

        @SuppressWarnings("deprecation")
        SQLContext sqlContext = new SQLContext(sc);
        SparkSession spark = SparkSession
                .builder()
                .appName("MLtree")
                .getOrCreate();

        Dataset<Row> data = spark
                .read()
                .option("header", true)
                .format("csv")
                .load("CO_cleaned.csv");

        Dataset<Row> rawdata = data.select("driver_age","driver_race","driver_gender","search_conducted");


        //delete null value rows
        Dataset<Row> cleanData = rawdata.filter((FilterFunction<Row>) row -> !row.anyNull());

        // Index labels, adding metadata to the label column.
        // Fit on whole dataset to include all labels in index.
        StringIndexerModel labelIndexer = new StringIndexer()
                .setInputCol("search_conducted")
                .setOutputCol("labelCol")
                .fit(cleanData);
        Dataset<Row> clean2 = labelIndexer.transform(cleanData);
        //index race column
        StringIndexerModel indexRace = new StringIndexer()
                .setInputCol("driver_race")
                .setOutputCol("raceIndex")
                .fit(cleanData);
        Dataset<Row> indexedRace = indexRace.transform(clean2);

      //convert gender column
        StringIndexerModel indexGender = new StringIndexer()
                .setInputCol("driver_gender")
                .setOutputCol("genIndex")
                .fit(indexedRace);
        Dataset<Row> indexedGender = indexGender.transform(indexedRace);

//        //convert age
        Dataset<Row> indexedAge = indexedGender.withColumn("driver_age",
                indexedGender.col("driver_age").cast("Double"));
        indexedAge.printSchema();


        double[] split = {0.0,20.0,50.0,100.0};

        //Dataset<Row> indexed3 = index3.transform(indexed2);
        indexedAge.show();
        indexedAge.printSchema();
        Bucketizer bucketizer = new Bucketizer()
                .setInputCol("driver_age")
                .setOutputCol("ageIndex")
                .setSplits(split);
        Dataset<Row> bucketed = bucketizer.transform(indexedAge);
        bucketed.show();

        //convert to vector
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"raceIndex","ageIndex","genIndex"})
                .setOutputCol("features");
        Dataset<Row> finalData = assembler.transform(bucketed);

        VectorIndexerModel featureIndexer = new VectorIndexer()
                .setInputCol("features")
                .setOutputCol("indexedFeatures")
                .setMaxCategories(6) // features with > 4 distinct values are treated as continuous.
                .fit(finalData);

        Row r = ChiSquareTest.test(finalData, "features", "labelCol").head();
        System.out.println("pValues: " + r.get(0).toString());
        System.out.println("degreesOfFreedom: " + r.getList(1).toString());
        System.out.println("statistics: " + r.get(2).toString());


    }
}