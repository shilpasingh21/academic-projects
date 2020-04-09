/*****Author: Shilpa Singh ***/

package com.review.rating.prediction;

import static org.apache.spark.sql.functions.callUDF;
import static org.apache.spark.sql.functions.col;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.ml.feature.RegexTokenizer;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.simple.Sentence;
import edu.stanford.nlp.trees.Tree;
import scala.collection.mutable.WrappedArray;

public class SentimentFeatureExtraction {

	private static Logger logger = Logger.getLogger(SentimentFeatureExtraction.class);

	public static void main(String[] args) throws IOException {

		System.setProperty("hadoop.home.dir", "C:\\Users\\shilp\\winutils");

		/* Read the parameter from the command line */

		if (args.length < 5) {
			System.err.println(
					"Usage: SentimentFeatureExtraction <business datafile in hadoop > <review datafile_in_hadoop> <nounoutdir> <adjoutdir> <nounadjoutdir>");
			System.exit(1);
		}

		String businessfile = args[0];
		String reviewfile = args[1];
		String nounoutdir = args[2];
		String adjoutdir = args[3];
		String nounadjoutdir = args[4];

		/* Instantiate a SparkSession with SqlContext */

		SparkSession spark = SparkSession.builder().appName("SentimentFeatureExtraction").getOrCreate();
		JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());

		/* Broadcasting NLP-helper class to other nodes */

		NlpHelper helper = new NlpHelper();
		Broadcast<NlpHelper> broadcasted = jsc.broadcast(helper);

		logger.info(" Nlp helper is broadcasted to the nodes ");

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

			private static final long serialVersionUID = 5206359647791544374L;

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

			private static final long serialVersionUID = 4497054206718610240L;

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

			private static final long serialVersionUID = 339617359138726759L;

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

		/* UDF for Sentiment score of each word */

		UDF1<WrappedArray<String>, Vector> udfsentiment = new UDF1<WrappedArray<String>, Vector>() {

			private static final long serialVersionUID = 3944616322978643792L;

			@Override
			public Vector call(WrappedArray<String> words) {

				if (!words.isEmpty()) {
					List<Double> scores = new ArrayList<Double>();
					scala.collection.Iterator<String> itr = words.iterator();
					while (itr.hasNext()) {
						String word = itr.next();
						Sentence sentence = new Sentence(word);
						StanfordCoreNLP pipeline = broadcasted.getValue().getOrCreateSentimentPipeline();
						Annotation annotation = pipeline.process(sentence.text());
						Tree tree = annotation.get(CoreAnnotations.SentencesAnnotation.class).get(0)
								.get(SentimentCoreAnnotations.SentimentAnnotatedTree.class);
						scores.add(new Double(RNNCoreAnnotations.getPredictedClass(tree)));
					}
					return Vectors.dense(scores.stream().mapToDouble(Double::doubleValue).toArray());
				} else {
					double[] emptyscores = { -1 };
					return Vectors.dense(emptyscores);
				}

			}
		};

		spark.sqlContext().udf().register("sentimentevaluator", udfsentiment, new VectorUDT());

		logger.info(" sentiment scorer udf is registered with spark ");

		Dataset<Row> noun_sentiment = nouns.withColumn("sentiment", callUDF("sentimentevaluator", col("nouncol")));

		Dataset<Row> adj_sentiment = adjectives.withColumn("sentiment",
				callUDF("sentimentevaluator", col("adjectivecol")));

		Dataset<Row> nounadj_sentiment = nounsadj.withColumn("sentiment",
				callUDF("sentimentevaluator", col("nounadjcol")));

		logger.info(" sentiment scorer evaluated for nouns,adjs,nouns+adjs ");

		/* Saving the data in libsvm format */

		VectorAssembler assembler = new VectorAssembler().setInputCols(new String[] { "sentiment" })
				.setOutputCol("features");

		Dataset<Row> nounlibsvm = assembler.transform(noun_sentiment).select(col("stars").as("label"), col("features"));
		Dataset<Row> adjlibsvm = assembler.transform(adj_sentiment).select(col("stars").as("label"), col("features"));
		Dataset<Row> nounadjlibsvm = assembler.transform(nounadj_sentiment).select(col("stars").as("label"),
				col("features"));

		nounlibsvm.repartition(1).write().format("libsvm").save(nounoutdir);
		adjlibsvm.repartition(1).write().format("libsvm").save(adjoutdir);
		nounadjlibsvm.repartition(1).write().format("libsvm").save(nounadjoutdir);

		logger.info(" sentiment data for nouns,adjs,nouns+adjs saved in libsvm format ");
		logger.info(" Business records with Restaurants category : " + business.count());
		logger.info(" Reviews records with Restaurants category : " + reviews.count());
		
		jsc.close();
	}

}
