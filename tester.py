import tensorflow as tf
import pickle

binary = True

with open('../test_images.pkl', 'rb') as f:
            test_data = pickle.load(f)

if binary:
    model = tf.keras.models.load_model("best_model_binary.keras")
    with open('../binary_test_labels.pkl', 'rb') as f:
            test_labels = pickle.load(f)
else:
    model = tf.keras.models.load_model("best_model_all_classes.keras")
    with open('../test_labels.pkl', 'rb') as f:
            test_labels = pickle.load(f)

test_loss, test_acc = model.evaluate(test_data, test_labels)
print(f"✅ Test Accuracy: {test_acc:.4f}")
print(f"📉 Test Loss: {test_loss:.4f}")
#print(f"Test Weighted F1 Score: {test_f1:.4f}")
print(model.summary())
print(model.count_params())
