
public class Main {

	public static void main(String[] args) throws Exception {

		System.out.print("Loading Data...");
		Dataset dataset = new Dataset("dataset/cleaned_olist_public_dataset_v2.csv");
		System.out.println("\t\t\tCompleted");

		System.out.print("Preparing Data...");
		int maxDaysDelivery = 13;
		dataset.selectRecords();
		dataset.selectAttributes();
		dataset.addOrderDelay(maxDaysDelivery);
		dataset.addOrderPurchaseDay();
		dataset.addProductPurchaseFrequency();
		dataset.mergeInfrequentValues("customer_city", 250);
		dataset.mergeInfrequentValues("customer_state", 1000);
		dataset.mergeInfrequentValues("product_category_name", 300);
		dataset.integrateTranslations("dataset/product_category_name_translation.csv");
		System.out.println("\t\tCompleted");

		System.out.print("Saving Data...");
		dataset.save("dataset/final_dataset.csv");
		System.out.println("\t\t\tCompleted");

		System.out.print("Learning decision tree...");
		int minObjectsPerLeaf = 25;
		DecisionTree tree = new DecisionTree(dataset, minObjectsPerLeaf);
		tree.visualize();
		System.out.println("\tCompleted");

		System.out.print("Performing cross validation...");
		int folds = 10;
		String evaluation = tree.runCrossValidation(dataset, folds);
		System.out.println("\tCompleted");

		System.out.print("Mining association rules...");
		int minSupport = 200;
		double minConfidence = 0.75;
		int maxGenericRules = 120;
		AssociationRules rules = new AssociationRules(dataset, minSupport, minConfidence, maxGenericRules);
		System.out.println("\tCompleted");

		System.out.print("Filtering rules...");
		rules.selectOrderDelayRules();
		rules.filterUselessRules();
		System.out.println("\t\tCompleted");

		System.out.println(evaluation);
		System.out.println(rules);
	}

}
