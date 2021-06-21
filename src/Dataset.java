import java.io.File;
import java.io.IOException;
import java.util.GregorianCalendar;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.CSVLoader;
import weka.core.converters.CSVSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;
import weka.filters.unsupervised.attribute.MergeInfrequentNominalValues;
import weka.filters.unsupervised.attribute.RemoveByName;
import weka.filters.unsupervised.attribute.RemoveType;
import weka.filters.unsupervised.attribute.RenameAttribute;
import weka.filters.unsupervised.attribute.RenameNominalValues;
import weka.filters.unsupervised.instance.RemoveFrequentValues;

class Dataset {

	private Instances data;

	Dataset(String filePath) throws Exception {
		DataSource source = new DataSource(filePath);
		((CSVLoader) source.getLoader()).setDateAttributes("7-10, 24-25");
		data = source.getDataSet();
	}

	Instances getInstances() {
		return data;
	}

	void selectRecords() throws Exception {
		data.deleteWithMissing(data.attribute("order_delivered_customer_date"));
		Attribute orderId = data.attribute("order_id");
		int oneRecordOrdersCount = data.attributeStats(orderId.index()).uniqueCount;
		RemoveFrequentValues compositeOrdersFilter = new RemoveFrequentValues();
		compositeOrdersFilter.setAttributeIndex(Integer.toString(orderId.index() + 1));
		compositeOrdersFilter.setInvertSelection(true);
		compositeOrdersFilter.setNumValues(data.numDistinctValues(orderId) - oneRecordOrdersCount);
		compositeOrdersFilter.setUseLeastValues(true);
		compositeOrdersFilter.setInputFormat(data);
		data = Filter.useFilter(data, compositeOrdersFilter);
	}

	void selectAttributes() throws Exception {
		RemoveByName filter = new RemoveByName();
		String toDelete = "order_id|order_status|order_sellers_qty|order_aproved_at|order_estimated_delivery_date|customer_id|review\\w*";
		filter.setExpression(toDelete);
		filter.setInputFormat(data);
		data = Filter.useFilter(data, filter);
	}

	void addOrderDelay(int maxDays) throws Exception {
		int millisecondsPerDay = 86400000;
		int maxWaiting = maxDays * millisecondsPerDay;
		Attribute purchaseDate = data.attribute("order_purchase_timestamp");
		Attribute deliveryDate = data.attribute("order_delivered_customer_date");
		Add newAttribute = new Add();
		newAttribute.setAttributeName("order_delay");
		newAttribute.setNominalLabels("true,false");
		newAttribute.setInputFormat(data);
		data = Filter.useFilter(data, newAttribute);
		data.setClassIndex(data.numAttributes() - 1);
		data.forEach(i -> i.setValue(data.classAttribute(),
				Boolean.toString(i.value(deliveryDate) - i.value(purchaseDate) > maxWaiting)));
		data.deleteAttributeAt(deliveryDate.index());
	}

	void addOrderPurchaseDay() throws Exception {
		int index = data.attribute("order_purchase_timestamp").index();
		Add newAttribute = new Add();
		newAttribute.setAttributeIndex(Integer.toString(index + 1));
		newAttribute.setAttributeName("order_purchase_day");
		newAttribute.setInputFormat(data);
		data = Filter.useFilter(data, newAttribute);
		GregorianCalendar calendar = new GregorianCalendar();
		Attribute orderPurchaseDay = data.attribute("order_purchase_day");
		Attribute purchaseDate = data.attribute("order_purchase_timestamp");
		for (Instance i : data) {
			calendar.setTimeInMillis((long) i.value(purchaseDate));
			i.setValue(orderPurchaseDay, calendar.get(GregorianCalendar.DAY_OF_YEAR));
		}
		data.deleteAttributeAt(purchaseDate.index());
	}

	void addProductPurchaseFrequency() throws Exception {
		int index = data.attribute("product_id").index();
		int productCount[] = data.attributeStats(index).nominalCounts;
		Add newAttribute = new Add();
		newAttribute.setAttributeIndex(Integer.toString(index + 1));
		newAttribute.setAttributeName("product_purchase_frequency");
		newAttribute.setInputFormat(data);
		data = Filter.useFilter(data, newAttribute);
		Attribute purchaseFrequency = data.attribute("product_purchase_frequency");
		Attribute productId = data.attribute("product_id");
		data.forEach(
				i -> i.setValue(purchaseFrequency, productCount[productId.indexOfValue(i.stringValue(productId))]));
		data.deleteAttributeAt(productId.index());
	}

	void mergeInfrequentValues(String attributeName, int minFrequency) throws Exception {
		int index = data.attribute(attributeName).index();
		MergeInfrequentNominalValues merge = new MergeInfrequentNominalValues();
		merge.setAttributeIndices(Integer.toString(index + 1));
		merge.setMinimumFrequency(minFrequency);
		merge.setUseShortIDs(true);
		merge.setInputFormat(data);
		data = Filter.useFilter(data, merge);
		RenameAttribute oldAttributeName = new RenameAttribute();
		oldAttributeName.setAttributeIndices(Integer.toString(index + 1));
		oldAttributeName.setReplace(attributeName);
		oldAttributeName.setInputFormat(data);
		data = Filter.useFilter(data, oldAttributeName);
		renameValues(attributeName, data.attribute(attributeName).value(0) + ":other");
	}

	private void renameValues(String attributeName, String replacements) throws Exception {
		int index = data.attribute(attributeName).index();
		RenameNominalValues newName = new RenameNominalValues();
		newName.setSelectedAttributes(Integer.toString(index + 1));
		newName.setValueReplacements(replacements);
		newName.setInputFormat(data);
		data = Filter.useFilter(data, newName);
	}

	void integrateTranslations(String filePath) throws Exception {
		DataSource source = new DataSource(filePath);
		Instances translations = source.getDataSet();
		StringBuilder replacements = new StringBuilder();
		translations.forEach(i -> replacements.append(i.stringValue(0) + ":" + i.stringValue(1) + ","));
		renameValues("product_category_name", replacements.toString());
	}

	void save(String path) throws IOException {
		CSVSaver saver = new CSVSaver();
		saver.setInstances(data);
		saver.setFile(new File(path));
		saver.writeBatch();
	}

	Instances filterNumericAttributes() throws Exception {
		RemoveType filter = new RemoveType();
		filter.setAttributeType(new SelectedTag("Delete numeric attributes", RemoveType.TAGS_ATTRIBUTETYPE));
		filter.setInputFormat(data);
		return Filter.useFilter(data, filter);
	}

}
