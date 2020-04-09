/*****Author: Shilpa Singh ***/

package com.review.rating.prediction;

import static org.apache.spark.sql.functions.callUDF;
import static org.apache.spark.sql.functions.col;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;
import org.apache.spark.ml.feature.CountVectorizer;
import org.apache.spark.ml.feature.CountVectorizerModel;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.IDFModel;
import org.apache.spark.ml.feature.RegexTokenizer;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;

import edu.stanford.nlp.simple.Sentence;
import scala.collection.mutable.WrappedArray;

public class TFIDFFeatureExtraction {

	private static Logger logger = Logger.getLogger(TFIDFFeatureExtraction.class);

	public static void main(String[] args) throws IOException {

		/* Read the parameter from the command line */

		if (args.length < 5) {
			System.err.println(
					"Usage: TFIDFFeatureExtraction <business datafile in hadoop > <review datafile_in_hadoop> <ldanounoutdir> <ldaadjoutdir> <ldanounadjoutdir>");
			System.exit(1);
		}

		String businessfile = args[0];
		String reviewfile = args[1];
		String nounoutdir = args[2];
		String adjoutdir = args[3];
		String nounadjoutdir = args[4];

		/* Instantiate a SparkSession with SqlContext */

		SparkSession spark = SparkSession.builder().appName("TFIdfFeatureExtraction").getOrCreate();

		/* Creating reviews data-frame for Restaurants business category */

		Dataset<Row> business = spark.read().json(businessfile).filter(col("categories").contains("Restaurants"));
		Dataset<Row> reviews = spark.read().json(reviewfile);
		Dataset<Row> joined = business
				.join(reviews, business.col("business_id").equalTo(reviews.col("business_id")), "inner")
				.select(business.col("business_id"), reviews.col("review_id"), reviews.col("text"),
						reviews.col("stars"));

		logger.info(" review file is read from Hdfs ");

		/* Text Tokenization */

		RegexTokenizer regexTokenizer = new RegexTokenizer().setInputCol("text").setOutputCol("words").setPattern("\\W")
				.setMinTokenLength(4);
		Dataset<Row> regexTokenized = regexTokenizer.transform(joined.na().fill("an"));

		logger.info(" Tokenization of text data completed ");

		/* Removing stopwords */

		StopWordsRemover stopwordsremover = new StopWordsRemover().setInputCol("words")
				.setOutputCol("filteredwithstopwords");
		Dataset<Row> filteredstopwords = stopwordsremover.transform(regexTokenized);

		logger.info(" Removing stopwords from text data completed ");

		/* UDF for selecting only nouns */

		UDF1<WrappedArray<String>, String[]> udfnouns = new UDF1<WrappedArray<String>, String[]>() {

			private static final long serialVersionUID = -8379903575815374437L;

			@Override
			public String[] call(WrappedArray<String> words) {
				List<String> tokens = new ArrayList<String>();
				if (!words.isEmpty()) {
					scala.collection.Iterator<String> itr = words.iterator();
					while (itr.hasNext()) {
						String word = itr.next();
						tokens.add(word);
					}
					List<String> tags = new Sentence(tokens).posTags();
					List<String> output = new ArrayList<String>();
					for (int i = 0; i < tags.size(); i++) {
						if (tags.get(i).contains("NN")) {
							output.add(tokens.get(i));
						}
					}

					return output.stream().toArray(String[]::new);
				} else {
					List<String> output = new ArrayList<String>();
					output.add("na");
					return output.stream().toArray(String[]::new);
				}
			}
		};

		/* UDF for selecting only adjectives */

		UDF1<WrappedArray<String>, String[]> udfadjs = new UDF1<WrappedArray<String>, String[]>() {

			private static final long serialVersionUID = -6902547316989980732L;

			@Override
			public String[] call(WrappedArray<String> words) {
				List<String> tokens = new ArrayList<String>();
				if (!words.isEmpty()) {
					scala.collection.Iterator<String> itr = words.iterator();
					while (itr.hasNext()) {
						String word = itr.next();
						tokens.add(word);
					}
					List<String> tags = new Sentence(tokens).posTags();
					List<String> output = new ArrayList<String>();
					for (int i = 0; i < tags.size(); i++) {
						if (tags.get(i).contains("JJ")) {
							output.add(tokens.get(i));
						}
					}
					return output.stream().toArray(String[]::new);
				} else {
					List<String> output = new ArrayList<String>();
					output.add("na");
					return output.stream().toArray(String[]::new);
				}
			}
		};

		/* UDF for selecting nouns and adjectives */

		UDF1<WrappedArray<String>, String[]> udfnounsadjs = new UDF1<WrappedArray<String>, String[]>() {

			private static final long serialVersionUID = -4536306137522615589L;

			@Override
			public String[] call(WrappedArray<String> words) {
				List<String> tokens = new ArrayList<String>();
				if (!words.isEmpty()) {
					scala.collection.Iterator<String> itr = words.iterator();
					while (itr.hasNext()) {
						String word = itr.next();
						tokens.add(word);
					}
					List<String> tags = new Sentence(tokens).posTags();
					List<String> output = new ArrayList<String>();
					for (int i = 0; i < tags.size(); i++) {
						if (tags.get(i).contains("NN") || tags.get(i).contains("JJ")) {
							output.add(tokens.get(i));
						}
					}
					return output.stream().toArray(String[]::new);
				} else {
					List<String> output = new ArrayList<String>();
					output.add("na");
					return output.stream().toArray(String[]::new);
				}
			}
		};

		/* POStagging of the data and selecting {nouns,adjectives,nouns+adjectives} */

		spark.sqlContext().udf().register("nounselector", udfnouns, DataTypes.createArrayType(DataTypes.StringType));
		spark.sqlContext().udf().register("adjselector", udfadjs, DataTypes.createArrayType(DataTypes.StringType));
		spark.sqlContext().udf().register("nounadjselector", udfnounsadjs,
				DataTypes.createArrayType(DataTypes.StringType));

		logger.info(" Postagger udfs registered with spark ");

		Dataset<Row> nouns = filteredstopwords.withColumn("nouncol",
				callUDF("nounselector", col("filteredwithstopwords")));

		Dataset<Row> adjectives = filteredstopwords.withColumn("adjectivecol",
				callUDF("adjselector", col("filteredwithstopwords")));

		Dataset<Row> nounsadj = filteredstopwords.withColumn("nounadjcol",
				callUDF("nounadjselector", col("filteredwithstopwords")));

		logger.info(" Seperate datasets for nouns,adjectives,nouns+adjectives is created ");

		/* Vectorization of nounsdata */

		CountVectorizerModel nounvectormodel = new CountVectorizer().setInputCol("nouncol").setOutputCol("features")
				.fit(nouns);
		Dataset<Row> nounvector = nounvectormodel.transform(nouns).select(col("features").as("rawFeatures"),
				col("stars"));

		/* Vectorization of adjsdata */

		CountVectorizerModel adjvectormodel = new CountVectorizer().setInputCol("adjectivecol").setOutputCol("features")
				.fit(adjectives);
		Dataset<Row> adjvector = adjvectormodel.transform(adjectives).select(col("features").as("rawFeatures"),
				col("stars"));

		/* Vectorization of nouns+adjs data */

		CountVectorizerModel nounadjvectormodel = new CountVectorizer().setInputCol("nounadjcol")
				.setOutputCol("features").fit(nounsadj);
		Dataset<Row> nounadjvector = nounadjvectormodel.transform(nounsadj).select(col("features").as("rawFeatures"),
				col("stars"));

		logger.info(" Word countvector of nouns,adjectives,nouns+adjectives created");

		/*
		 * Generating tf-idf model of reviews text in libsvm format IDF is an Estimator
		 * which is fit on countvectorizer and produces an IDFModel
		 */

		IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features");

		IDFModel nounidfModel = idf.fit(nounvector);
		IDFModel adjidfModel = idf.fit(adjvector);
		IDFModel nounadjidfModel = idf.fit(nounadjvector);

		logger.info(" tf-idf model created for the nouns, adjectives, nouns+adjectives");

		/*
		 * The IDFModel takes feature vectors (generally created from HashingTF or
		 * CountVectorizer) and scales each column
		 */

		Dataset<Row> nounrescaledData = nounidfModel.transform(nounvector).select(col("stars").as("label"),
				col("features"));
		Dataset<Row> adjrescaledData = adjidfModel.transform(adjvector).select(col("stars").as("label"),
				col("features"));
		Dataset<Row> nounadjrescaledData = nounadjidfModel.transform(nounadjvector).select(col("stars").as("label"),
				col("features"));

		logger.info(" Transformation of data to tf-idf form for nouns, adjectives, nouns+adjectives completed");

		nounrescaledData.repartition(1).write().format("libsvm").save(nounoutdir);
		adjrescaledData.repartition(1).write().format("libsvm").save(adjoutdir);
		nounadjrescaledData.repartition(1).write().format("libsvm").save(nounadjoutdir);

		logger.info(" libsvm data for tf-idf model saved in hdfs ");

	}

}
