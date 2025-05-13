import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.trees.J48;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import java.io.File;
import java.text.DecimalFormat;
import java.util.Random;

public class MovieBoxOfficePredictor {
    public static void main(String[] args) {
        try {
            // Initialize CSV loader and load dataset
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File("movies_dataset.csv"));
            Instances dataset = loader.getDataSet();
            dataset.setClassIndex(dataset.numAttributes() - 1); // Revenue is the target

            // Split dataset: 75% train, 25% test
            dataset.randomize(new Random(42));
            int trainSize = (int) Math.round(dataset.numInstances() * 0.75);
            int testSize = dataset.numInstances() - trainSize;
            Instances trainData = new Instances(dataset, 0, trainSize);
            Instances testData = new Instances(dataset, trainSize, testSize);

            // Define models
            Classifier[] classifiers = {
                new LinearRegression(),
                new J48(), // CART implementation
                new MultilayerPerceptron(), // ANN
                new IBk(3) // KNN with k=3
            };
            String[] classifierNames = {"Linear Regression", "Decision Tree (CART)", "Neural Network (ANN)", "K-Nearest Neighbors (KNN)"};

            // Format for output
            DecimalFormat df = new DecimalFormat("#.####");
            System.out.println("=== Movie Box Office Prediction Results ===");

            // Train and evaluate each model
            for (int i = 0; i < classifiers.length; i++) {
                Classifier model = classifiers[i];
                model.buildClassifier(trainData);
                
                Evaluation eval = new Evaluation(trainData);
                eval.evaluateModel(model, testData);
                
                System.out.println("\nModel: " + classifierNames[i]);
                System.out.println("RMSE: " + df.format(eval.rootMeanSquaredError()));
                System.out.println("MAE: " + df.format(eval.meanAbsoluteError()));
            }

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}