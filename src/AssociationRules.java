import java.text.DecimalFormat;
import java.util.HashSet;
import java.util.List;
import java.util.stream.Collectors;

import weka.associations.Apriori;
import weka.associations.ItemSet;
import weka.core.Instances;
import weka.core.SelectedTag;

class AssociationRules {

	private Instances data;

	private List<ItemSet> premises;

	private List<ItemSet> consequences;

	AssociationRules(Dataset dataset, int minSupport, double minConfidence, int maxGenericRules) throws Exception {
		data = dataset.filterNumericAttributes();
		Apriori rulesMiner = new Apriori();
		rulesMiner.setCar(true);
		rulesMiner.setLowerBoundMinSupport((float) minSupport / dataset.getInstances().size());
		rulesMiner.setMetricType(new SelectedTag("Confidence", Apriori.TAGS_SELECTION));
		rulesMiner.setMinMetric(minConfidence);
		rulesMiner.setNumRules(maxGenericRules);
		rulesMiner.buildAssociations(data);
		List<Object> rules[] = rulesMiner.getAllTheRules();
		premises = rules[0].stream().map(e -> (ItemSet) e).collect(Collectors.toList());
		consequences = rules[1].stream().map(e -> (ItemSet) e).collect(Collectors.toList());
	}

	void selectOrderDelayRules() {
		for (int i = 0; i < consequences.size(); i++) {
			int classValueIndex = consequences.get(i).itemAt(0);
			if (classValueIndex == data.classAttribute().indexOfValue("false")) {
				consequences.remove(i);
				premises.remove(i);
				i--;
			}
		}
	}

	void filterUselessRules() {
		for (int i = 0; i < premises.size(); i++)
			for (int j = 0; j < premises.size(); j++)
				if (i != j && isUseless(premises.get(i), premises.get(j))) {
					premises.remove(i);
					consequences.remove(i);
					i--;
					break;
				}
	}

	private boolean isUseless(ItemSet a, ItemSet b) {
		if (a.support() != b.support())
			return false;
		HashSet<String> setA = new HashSet<>();
		HashSet<String> setB = new HashSet<>();
		for (int i = 0; i < a.getItems().length; i++) {
			if (a.itemAt(i) >= 0)
				setA.add(i + ":" + a.itemAt(i));
			if (b.itemAt(i) >= 0)
				setB.add(i + ":" + b.itemAt(i));
		}
		return setA.containsAll(setB);
	}

	public String toString() {
		String rules = "\nBest Class Association Rules:\n";
		for (int i = 0; i < premises.size(); i++) {
			rules += "\n" + (i + 1) + ". ";
			int valuesIndex[] = premises.get(i).getItems();
			for (int j = 0; j < valuesIndex.length; j++)
				if (valuesIndex[j] >= 0)
					rules += data.attribute(j).name() + "=" + data.attribute(j).value(valuesIndex[j]) + " ";
			rules += "(" + premises.get(i).support() + ") ==> " + data.classAttribute().name() + "=";
			rules += data.classAttribute().value(consequences.get(i).itemAt(0));
			rules += " (" + consequences.get(i).support() + ")";
			DecimalFormat df = new DecimalFormat();
			df.setMaximumFractionDigits(2);
			rules += "  Conf:" + df.format((float) consequences.get(i).support() / premises.get(i).support());
		}
		return rules;
	}

}
