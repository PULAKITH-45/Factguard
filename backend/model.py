from transformers import pipeline

print("Loading Fake News Model...")

# 🔹 Load working public model
try:
    classifier = pipeline(
        "text-classification",
        model="jy46604790/Fake-News-Bert-Detect"
    )
    print("Model loaded successfully!")
except Exception as e:
    print("Model load failed:", e)
    classifier = None


# 🔹 Prediction function
def get_prediction(text: str):

    # If model not loaded
    if classifier is None:
        return {
            "prediction": "NOT VERIFIED",
            "reason": "AI Model not available.",
            "confidence": 0.0
        }

    try:
        result = classifier(text)[0]

        label = result["label"]
        score = result["score"]

        # 🔥 Correct mapping
        if "FAKE" in label.upper():
            prediction = "NOT VERIFIED"
            reason = "Model detected possible fake news"
        else:
            prediction = "REAL"
            reason = "Model detected reliable information"

        return {
            "prediction": prediction,
            "reason": reason,
            "confidence": round(score * 100, 2)
        }

    except Exception as e:
        print("Prediction error:", e)
        return {
            "prediction": "NOT VERIFIED",
            "reason": "Error during prediction",
            "confidence": 0.0
        }


# 🔹 Dummy placeholders (safe to keep)
class DummyModel:
    def eval(self): pass
    def to(self, device): pass

model = DummyModel()
tokenizer = None
bert_fakereal_model = None
bert_tokenizer = None