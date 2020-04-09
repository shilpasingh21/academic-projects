
/*****Author: Shilpa Singh ***/
package com.review.rating.prediction;

import static org.apache.spark.sql.functions.callUDF;
import static org.apache.spark.sql.functions.col;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;
import org.apache.spark.ml.clustering.DistributedLDAModel;
import org.apache.spark.ml.clustering.LDA;
import org.apache.spark.ml.feature.CountVectorizer;
import org.apache.spark.ml.feature.CountVectorizerModel;
import org.apache.spark.ml.feature.RegexTokenizer;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;

import edu.stanford.nlp.simple.Sentence;
import scala.collection.mutable.WrappedArray;

public class LDAFeatureExtraction {

	private static Logger logger = Logger.getLogger(LDAFeatureExtraction.class);

	public static void main(String[] args) throws IOException {

		/* Read the parameter from the command line */

		if (args.length < 5) {
			System.err.println(
					"Usage: LDAFeatureExtraction <business datafile in hadoop > <review datafile_in_hadoop> <ldanounoutdir> <ldaadjoutdir> <ldanounadjoutdir>");
			System.exit(1);
		}

		String businessfile = args[0];
		String reviewfile = args[1];
		String nounoutdir = args[2];
		String adjoutdir = args[3];
		String nounadjoutdir = args[4];

		/* Instantiate a SparkSession with SqlContext */

		SparkSession spark = SparkSession.builder().appName("LDAFeatureExtraction").getOrCreate();

		/* Creating reviews data-frame for Restaurants business category */

		Dataset<Row> business = spark.read().json(businessfile).filter(col("categories").contains("Restaurants"));
		Dataset<Row> reviews = spark.read().json(reviewfile);
		Dataset<Row> joined = business
				.join(reviews, business.col("business_id").equalTo(reviews.col("business_id")), "inner")
				.select(business.col("business_id"), reviews.col("review_id"), reviews.col("text"),
						reviews.col("stars"));

		logger.info(" Review file is read from Hdfs ");

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
		Dataset<Row> nounvector = nounvectormodel.transform(nouns);

		/* Vectorization of adjsdata */

		CountVectorizerModel adjvectormodel = new CountVectorizer().setInputCol("adjectivecol").setOutputCol("features")
				.fit(adjectives);
		Dataset<Row> adjvector = adjvectormodel.transform(adjectives);

		/* Vectorization of nouns+adjs data */

		CountVectorizerModel nounadjvectormodel = new CountVectorizer().setInputCol("nounadjcol")
				.setOutputCol("features").fit(nounsadj);
		Dataset<Row> nounadjvector = nounadjvectormodel.transform(nounsadj);

		logger.info(" Vectorization of nouns,adjectives,nouns+adjectives completed");

		/*
		 * Instantiating EM(Expectation Minimization) LDA model with the k-20 topics and
		 * 100 iterations for the above 3 datasets
		 */

		LDA lda = new LDA().setSeed(80).setMaxIter(100).setK(20).setOptimizer("em").setDocConcentration(3.5)
				.setTopicConcentration(1.5);

		DistributedLDAModel nounmodel = (DistributedLDAModel) lda.fit(nounvector);
		Dataset<Row> nountransformed = nounmodel.transform(nounvector);

		logger.info(" LDA Topic model created for nouns ");

		DistributedLDAModel adjmodel = (DistributedLDAModel) lda.fit(adjvector);
		Dataset<Row> adjtransformed = adjmodel.transform(adjvector);

		logger.info(" LDA Topic model created for adjectives ");

		DistributedLDAModel nounadjmodel = (DistributedLDAModel) lda.fit(nounadjvector);
		Dataset<Row> nounadjtransformed = nounadjmodel.transform(nounadjvector);

		logger.info(" LDA Topic model created for nouns + adjectives ");

		/* Saving the data in libsvm format */

		Dataset<Row> nounlibsvm = nountransformed.select(col("stars").as("label"),
				col("topicDistribution").as("features"));
		Dataset<Row> adjlibsvm = adjtransformed.select(col("stars").as("label"),
				col("topicDistribution").as("features"));
		Dataset<Row> nounadjlibsvm = nounadjtransformed.select(col("stars").as("label"),
				col("topicDistribution").as("features"));

		nounlibsvm.repartition(1).write().format("libsvm").save(nounoutdir);
		adjlibsvm.repartition(1).write().format("libsvm").save(adjoutdir);
		nounadjlibsvm.repartition(1).write().format("libsvm").save(nounadjoutdir);

		logger.info(" Libsvm file for LDA model saveds ");

	}

}
