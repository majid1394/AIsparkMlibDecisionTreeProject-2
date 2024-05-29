

import au.com.bytecode.opencsv.CSVReader;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.util.MLUtils;
import scala.Tuple2;

import java.io.FileReader;
import java.util.HashMap;
import java.util.Map;

        /*The example below demonstrates how to load a LIBSVM data file,
                parse it as an RDD of LabeledPoint and then perform regression
                using a decision tree with variance as an impurity measure and a maximum tree depth of 5.
                The Mean Squared Error (MSE) is computed at the end to evaluate goodness of fit.*/

public class spark_MLip_DecisionTreeModel {
    public static void main(String[] args) {
        // if you need to convert your data
        DataConverter();
        SparkConf sparkConf = new SparkConf().setAppName("JavaDecisionTreeRegressionExample").setMaster("local");
        JavaSparkContext jsc = new JavaSparkContext(sparkConf);

// Load and parse the data file.
        /*String datapath = "data/mllib/sample_libsvm_data.txt";*/
        /*String datapath = "C:\\Users\\majid\\Downloads\\dataset\\bank+marketing\\bank\\majid.txt";*/
        String datapath = "src/main/resources/static/majid.txt";
        JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(jsc.sc(), datapath).toJavaRDD();
// Split the data into training and test sets (30% held out for testing)
        JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.7, 0.3});
        JavaRDD<LabeledPoint> trainingData = splits[0];
        JavaRDD<LabeledPoint> testData = splits[1];

// Set parameters.
// Empty categoricalFeaturesInfo indicates all features are continuous.
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
        String impurity = "variance";
        int maxDepth = 5;
        int maxBins = 32;

// Train a DecisionTree model.
        DecisionTreeModel model = DecisionTree.trainRegressor(trainingData,
                categoricalFeaturesInfo, impurity, maxDepth, maxBins);

// Evaluate model on test instances and compute test error
        JavaPairRDD<Double, Double> predictionAndLabel =
                testData.mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()));
        double testMSE = predictionAndLabel.mapToDouble(pl -> {
            double diff = pl._1() - pl._2();
            return diff * diff;
        }).mean();


        System.out.println("Test Mean Squared Error: " + testMSE);
        System.out.println("Learned regression tree model:\n" + model.toDebugString());

// Save and load model
        model.save(jsc.sc(), "target/tmp/myDecisionTreeRegressionModel");
        DecisionTreeModel sameModel = DecisionTreeModel
                .load(jsc.sc(), "target/tmp/myDecisionTreeRegressionModel");
    }
        public static void DataConverter() {
            try {
                CSVReader reader = new CSVReader(new FileReader("C:\\Users\\majid\\Downloads\\dataset\\bank+marketing\\bank\\bank-full.csv"));
                String[] nextLine;
                while ((nextLine = reader.readNext()) != null) {
                    // Process each line of the CSV file
                    for (String token : nextLine) {
                        System.out.print(token + " ");
                    }
                    System.out.println();
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
    }



}
