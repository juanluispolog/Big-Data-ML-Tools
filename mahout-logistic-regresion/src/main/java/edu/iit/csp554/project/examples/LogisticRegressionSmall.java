package edu.iit.csp554.project.examples;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.text.NumberFormat;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import org.apache.mahout.classifier.evaluation.Auc;
import org.apache.mahout.classifier.sgd.L1;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.vectorizer.encoders.ConstantValueEncoder;

public class LogisticRegressionSmall {

	private final int TRAIN_ROWS_NUM = 320;

	public static void main(final String[] args) {
		final long startTime = System.currentTimeMillis();

		final LogisticRegressionSmall logisticRegression = new LogisticRegressionSmall();

		// Load the input data
		final List<Observation> inputData = logisticRegression.parseInputFile("input/Alumns.csv");

		System.out.printf("Read file in: %d ms\n", (System.currentTimeMillis() - startTime));
		final long readTime = System.currentTimeMillis();

		// Train a model
		final OnlineLogisticRegression olr = logisticRegression.train(inputData);

		System.out.printf("Trained dataset in: %d ms\n", (System.currentTimeMillis() - readTime));

		// Test the model
		logisticRegression.testModel(olr, inputData);

		System.out.printf("Elapsed: %d ms\n", (System.currentTimeMillis() - startTime));
	}

	public List<Observation> parseInputFile(final String inputFile) {
		final List<Observation> result = new ArrayList<Observation>();
		BufferedReader br = null;
		String line = "";
		System.out.println("Reading input file...");
		try {
			// Load the file which contains training data
			br = new BufferedReader(new FileReader(new File(inputFile)));
			// Skip the first line which contains the header values
			line = br.readLine();
			// Prepare the observation data
			while ((line = br.readLine()) != null) {
				final String[] values = line.split(",");
				result.add(new Observation(values));
			}
		} catch (final FileNotFoundException e) {
			e.printStackTrace();
		} catch (final IOException e) {
			e.printStackTrace();
		} finally {
			if (br != null) {
				try {
					br.close();
				} catch (final IOException e) {
					e.printStackTrace();
				}
			}
		}
		return result;
	}

	public OnlineLogisticRegression train(final List<Observation> inputData) {
		final OnlineLogisticRegression olr = new OnlineLogisticRegression(2, 3, new L1());
		// Train the model using 30 passes
		for (int pass = 0; pass < 100; pass++) {
			for (int i = 0; i < TRAIN_ROWS_NUM; i++) {
				final Observation tObservation = inputData.get(i);
				olr.train(tObservation.getActual(), tObservation.getVector());
			}
			// Every 10 passes check the accuracy of the trained model
			if (pass % 10 == 0) {
				final Auc eval = new Auc(0.5);
				for (int i = 0; i < TRAIN_ROWS_NUM; i++) {
					final Observation tObservation = inputData.get(i);
					eval.add(tObservation.getActual(), olr.classifyScalar(tObservation.getVector()));
				}
				System.out.format("Pass: %2d, Learning rate: %2.4f, Accuracy: %2.4f\n", pass, olr.currentLearningRate(),
						eval.auc());
			}
		}
		return olr;
	}

	void testModel(final OnlineLogisticRegression olr, final List<Observation> inputData) {
		System.out.println("------------- Testing -------------");
		double firstClassSum = 0.f;

		float firstClassTrue = 0;
		float firstClassFalse = 0;
		float secondClassTrue = 0;
		float secondClassFalse = 0;

		for (int i = TRAIN_ROWS_NUM; i < inputData.size(); i++) {
			final Observation newObservation = inputData.get(i);
			final Vector result = olr.classifyFull(newObservation.getVector());
	
			if (result.get(0) >= 0.5 && inputData.get(i).getActual() == 0) {
				firstClassTrue += 1;
			}
			else if (result.get(0) >= 0.5 && inputData.get(i).getActual() == 1) {
				firstClassFalse += 1;
			}
			else if (result.get(0) < 0.5 && inputData.get(i).getActual() == 1) {
				secondClassTrue += 1;
			}
			else {
				secondClassFalse += 1;
			}

			firstClassSum += result.get(0);
		}

		final double avgFirstClass = firstClassSum / (inputData.size() - TRAIN_ROWS_NUM);

		double sensitivity = firstClassTrue / (firstClassTrue + secondClassTrue);
		double specificity = secondClassTrue / (secondClassTrue + firstClassTrue);
		double precission = firstClassTrue / (firstClassTrue + firstClassFalse);
		double accuracy = (firstClassTrue + secondClassTrue) / (firstClassTrue + firstClassFalse
		 	+ secondClassTrue + secondClassFalse);
		double testError = 1 - accuracy;

		System.out.println("\n--------------------------------");

		System.out.printf("Statistics:\n\tSensitivity: %.3f\n\tSpecificity: %.3f\n\tPrecission: %.3f\n\tAccuracy: %.3f\n\tTest error: %.3f\n"
		, sensitivity, specificity, precission, accuracy, testError);

		System.out.printf("Confusion matrix:\n\t%d   %d\n\t%d   %d\n", (int) firstClassTrue, (int) firstClassFalse, (int) secondClassTrue, (int) secondClassFalse);

		System.out.format("Average Probability not admitted (0) = %.3f\n", avgFirstClass);
		System.out.format("Average Probability admitted (1)     = %.3f\n", 1 - avgFirstClass);
	}

	private class Observation {
		private final DenseVector vector = new DenseVector(3);
		private int actual;

		public Observation(final String[] values) {
			final ConstantValueEncoder interceptEncoder = new ConstantValueEncoder("intercept");

			interceptEncoder.addToVector("1", vector);
			for (int i = 1; i < values.length; i++) {
				vector.set(i - 1, Double.valueOf(values[i]));
			}

			final NumberFormat f = NumberFormat.getNumberInstance(Locale.US);
			try {
				final Number n = f.parse(values[0]);

				this.actual = n.intValue();
			} catch (final ParseException e) {
				e.printStackTrace();
			}			
		}

		public Vector getVector() {
			return vector;
		}

		public int getActual() {
			return actual;
		}
	}
}