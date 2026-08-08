from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("models/legal_bert_final")
print(list(model.config.id2label.values()))