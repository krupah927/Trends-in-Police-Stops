import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.feature.Bucketizer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.sum;
import static org.apache.spark.sql.functions.year;

public class NewTrafficStops {
    public static void main(String args[]){
        JavaSparkContext sparkContext
                = new JavaSparkContext(new SparkConf()
                .setAppName("trafficstops")
                .setMaster("local[*]"));

        SparkSession sparkSession = SparkSession
                .builder()
                .appName("trafficstops")
                .getOrCreate();
        String file = "/home/rutuja/CS226/data/FL_cleaned.csv";

        Dataset<Row> stateData = sparkSession
                .read()
                .option("header", true)
                .option("inferSchema","true")
                .format("csv")
                .load(file);

        Dataset<Row> censusData = sparkSession
                .read()
                .format("csv")
                .option("header","true")
                .option("inferSchema","true")
                .load("/home/rutuja/IdeaProjects/NewTrafficStops/census.csv");


        Dataset<Row> stopData= stateData.select("id","state","stop_date","driver_age","driver_gender","driver_race");

        Dataset<Row> population= censusData.select("state","year","age","gender","race","count")
                .withColumnRenamed("count","pop")
                .withColumnRenamed("year","popYear")
                .withColumnRenamed("state", "censusState");



        Dataset<Row> stopDataWithProperAge = stopData.withColumn("driver_age"
                ,stopData.col("driver_age").cast("Double"));


        Dataset<Row> populationWithProperAge = population.withColumn("age",
                population.col("age").cast("Double"));



        double[] split = {0.0,20.0,50.0,100.0};



        //indexedAge.printSchema();

        Bucketizer bucketizer = new Bucketizer()
                .setInputCol("driver_age")
                .setOutputCol("driverAge")
                .setSplits(split);

        Dataset<Row> bucketedStopData = bucketizer.transform(stopDataWithProperAge);



        Bucketizer PopBucketizer = new Bucketizer()
                .setInputCol("age")
                .setOutputCol("AgePop")
                .setSplits(split);

        Dataset<Row> bucketedPopulation = PopBucketizer.transform(populationWithProperAge);



        Dataset<Row> genStopData =
                bucketedStopData.select("state","stop_date","driver_gender")
                        .groupBy(year(col("stop_date")),col("driver_gender"),col("state"))
                        .count()
                        .withColumnRenamed("year(stop_date)","yearStop")
                        .withColumnRenamed("count","Count");



        //census data
        Dataset<Row> genPop=
                bucketedPopulation.select("censusState","popYear","gender","pop")
                        .groupBy(col("censusState"),col("popYear"),col("gender"))
                        .agg(sum("pop"))
                        .withColumnRenamed("sum(pop)","total")
                        .sort("total");



        Dataset<Row> joinedGender =
                genStopData
                        .join(genPop,(genStopData.col("state").equalTo(genPop.col("censusState")))
                                .and(genStopData.col("driver_gender").equalTo(genPop.col("gender")))
                                .and(genStopData.col("yearStop").equalTo(genPop.col("popYear")))
                        );
        //.createOrReplaceTempView("join");

        //Dataset<Row> b =
        joinedGender.select("yearStop", "driver_gender", "state", "Count", "total")
                .groupBy("yearStop","driver_gender","state","Count","total")
                .agg(col("Count").divide(col("total")))
                .withColumnRenamed("(Count / total)","percentage")
                .coalesce(1)
                .write().csv("/home/rutuja/IdeaProjects/NewTrafficStops/FLgenderPerc");

        //---------------------------------------------------------------------------------------------------

        Dataset<Row> raceStopData =
                bucketedStopData.select("state","stop_date","driver_race")
                        .groupBy(year(col("stop_date")),col("driver_race"),col("state"))
                        .count()
                        .withColumnRenamed("year(stop_date)","yearStop")
                        .withColumnRenamed("count","Count");



        //census data
        Dataset<Row> racePop=
                bucketedPopulation.select("censusState","popYear","race","pop")
                        .groupBy(col("censusState"),col("popYear"),col("race"))
                        .agg(sum("pop"))
                        .withColumnRenamed("sum(pop)","total")
                        .sort("total");



        Dataset<Row> joinedRace =
                raceStopData
                        .join(racePop,(raceStopData.col("state").equalTo(racePop.col("censusState")))
                                .and(raceStopData.col("driver_race").equalTo(racePop.col("race")))
                                .and(raceStopData.col("yearStop").equalTo(racePop.col("popYear")))
                        );
        //.createOrReplaceTempView("join");

        //Dataset<Row> b =
        joinedRace.select("yearStop", "driver_race", "state", "Count", "total")
                .groupBy("yearStop","driver_race","state","Count","total")
                .agg(col("Count").divide(col("total")))
                .withColumnRenamed("(Count / total)","percentage")
                .coalesce(1)
                .write().csv("/home/rutuja/IdeaProjects/NewTrafficStops/FLracePerc");

        //-------------------------------------------------------------------------------------------------

        Dataset<Row> ageStopData =
                bucketedStopData.select("state","stop_date","driverAge")
                        .groupBy(year(col("stop_date")),col("driverAge"),col("state"))
                        .count()
                        .withColumnRenamed("year(stop_date)","yearStop")
                        .withColumnRenamed("count","Count");



        //census data
        Dataset<Row> agePop=
                bucketedPopulation.select("censusState","popYear","AgePop","pop")
                        .groupBy(col("censusState"),col("popYear"),col("AgePop"))
                        .agg(sum("pop"))
                        .withColumnRenamed("sum(pop)","total")
                        .sort("total");



        Dataset<Row> joinedAge =
                ageStopData
                        .join(agePop,(ageStopData.col("state").equalTo(agePop.col("censusState")))
                                .and(ageStopData.col("driverAge").equalTo(agePop.col("AgePop")))
                                .and(ageStopData.col("yearStop").equalTo(agePop.col("popYear")))
                        );
        //.createOrReplaceTempView("join");

        //Dataset<Row> b =
        joinedAge.select("yearStop", "driverAge", "state", "Count", "total")
                .groupBy("yearStop","driverAge","state","Count","total")
                .agg(col("Count").divide(col("total")))
                .withColumnRenamed("(Count / total)","percentage")
                .coalesce(1)
                .write().csv("/home/rutuja/IdeaProjects/NewTrafficStops/FLagePerc");









    }
}