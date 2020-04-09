/*****Author: Shilpa Singh ***/

package com.review.rating.prediction;


import java.io.Serializable;
import java.util.Properties;

import edu.stanford.nlp.pipeline.StanfordCoreNLP;

public class NlpHelper implements Serializable {

    private static final long serialVersionUID = -6335824457866202676L;
	private transient StanfordCoreNLP pipeline;

	public StanfordCoreNLP getOrCreateSentimentPipeline() {
		if (pipeline == null) {
			Properties props = new Properties();
			props.put("annotators", "tokenize, ssplit, parse, pos, sentiment");
			pipeline = new StanfordCoreNLP(props);
		}
		return pipeline;
	}

}
