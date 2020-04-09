/*****Author: Shilpa Singh ***/

package com.review.rating.prediction;


import java.io.File;
import java.io.IOException;

import org.apache.commons.io.FileUtils;
import org.apache.log4j.Logger;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.json.CDL;
import org.json.JSONArray;
import org.json.JSONObject;

public class SparkLinearRegression {

	private static Logger logger = Logger.getLogger(SparkLinearRegression.class);

	public static void main(String[] args) throws IOException {
		
		/* Read the parameter from the command line */

		if (args.length < 5) {
			System.err.println(
					"Usage: SparkLinearRegression <libsvm datafile in hadoop > <modeloutputdir in local filesystem> <statsfilename> <modeltype> <postype>");
			System.exit(1);
		}

		String datafile = args[0];
		String modeloutputdir = args[1];
		String statsfilename = args[2];
		String modeltype = args[3];
		String postype = args[4];
		File statsfile = new File(statsfilename);
		
		logger.info("Command line parameters read from input");
		
		/* Instantiate a SparkSession with SqlContext */

		SparkSession spark = SparkSession.builder().appName("SparkLinearRegression")
				.getOrCreate();
		
		logger.info("Instantiated Spark Session");

		/* Reading libsvm feature data from hdfs */

		Dataset<Row> data = spark.read().format("libsvm").load(datafile);
		
		logger.info("Libsvm data read from Hdfs");
		
        /* Split the data in test and training */
		
		Dataset<Row>[] splits = data.randomSplit(new double[] { 0.8, 0.2 });
		Dataset<Row> trainingData = splits[0];
		Dataset<Row> testData = splits[1];
		
		logger.info("Data is split into training and test set");
		
		/*Instantiate a linear regression model */
		
		LinearRegression lr = new LinearRegression()
				  .setMaxIter(100)
				  .setRegParam(0.3)
				  .setElasticNetParam(0.8);
		
		/* Fit the model in training set */
		
		LinearRegressionModel lrModel = lr.fit(trainingData);
		LinearRegressionTrainingSummary trainingSummary = lrModel.summary();
		
		logger.info("Regression model instantiated and fitted over training data");

		/* Make predictions on test data */

		Dataset<Row> predictions = lrModel.transform(testData);
		
		logger.info("Regression predictions completed on test data");
		
		/* Create a new JSONObject to save the statistics for the regression model */

		JSONArray statsjson = new JSONArray();
		JSONObject statjson = new JSONObject();
		
		statjson.put("Coefficients" , lrModel.coefficients());
		statjson.put("Intercept" , lrModel.intercept());
		statjson.put("RMSE" , trainingSummary.rootMeanSquaredError());
		statjson.put("MAE" , trainingSummary.meanAbsoluteError());
		statjson.put("R2" , trainingSummary.r2());
		statjson.put("modeltype" , modeltype);
		statjson.put("postype" , postype);
		statsjson.put(statjson);
		
		/* Write the statistics json as a CSV file to the local filesystem */

		String csv = CDL.toString(statsjson);
		FileUtils.writeStringToFile(statsfile, csv);
		
		logger.info("Regression statistics saved to the local filesystem");
		
		predictions.write().option("header", "true").json(modeloutputdir);
		
		logger.info("Predicted model saved to the local filesystem");
		
	}

}
