import java.awt.BorderLayout;
import java.util.Random;
import javax.swing.JFrame;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.J48;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;

class DecisionTree {

	private J48 tree;

	DecisionTree(Dataset dataset, int minObjPerLeaf) throws Exception {
		tree = new J48();
		tree.setBinarySplits(true);
		tree.setUnpruned(false);
		tree.setSubtreeRaising(true);
		tree.setConfidenceFactor(0.02f);
		tree.setMinNumObj(minObjPerLeaf);
		tree.buildClassifier(dataset.getInstances());
	}

	String runCrossValidation(Dataset dataset, int folds) throws Exception {
		Evaluation evaluation = new Evaluation(dataset.getInstances());
		evaluation.crossValidateModel(tree, dataset.getInstances(), folds, new Random(1));
		String results = "\n\n\nTree Evaluation:\n";
		results += evaluation.toSummaryString() + "\n";
		results += evaluation.toClassDetailsString() + "\n";
		results += evaluation.toMatrixString() + "\n";
		return results;
	}

	void visualize() throws Exception {
		TreeVisualizer visualizer = new TreeVisualizer(null, tree.graph(), new PlaceNode2());
		JFrame frame = new JFrame();
		frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		frame.setSize(1600, 1000);
		frame.getContentPane().setLayout(new BorderLayout());
		frame.getContentPane().add(visualizer, BorderLayout.CENTER);
		frame.setVisible(true);
		visualizer.fitToScreen();
	}

}
