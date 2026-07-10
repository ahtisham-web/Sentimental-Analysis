"""
Phase 12: Testing.
Runs the required test cases directly against the trained model + vectorizer
(no server needed) and prints pass/fail based on expected sentiment direction.
"""
import pickle
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "backend"))
from utils import preprocess_text

with open(BASE_DIR / "backend" / "model.pkl", "rb") as f:
    model = pickle.load(f)
with open(BASE_DIR / "backend" / "vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


def predict(text: str):
    if not text.strip():
        return "neutral (empty input handled)"
    cleaned = preprocess_text(text, method="lemmatize")
    vec = vectorizer.transform([cleaned])
    return model.predict(vec)[0]


TEST_CASES = [
    ("Positive sentence", "I absolutely love this product, it works great!", "positive"),
    ("Negative sentence", "This is the worst experience I've ever had, totally disappointed.", "negative"),
    ("Neutral sentence", "The package will arrive on Thursday at noon.", "neutral"),
    ("Empty input", "", None),
    ("Long paragraph", (
        "I was initially skeptical about trying this new service, but after using it "
        "for a full month, I have to say the experience has been fantastic. The support "
        "team responded quickly, the interface is intuitive, and I've genuinely enjoyed "
        "every interaction so far. I would recommend this to anyone looking for a reliable "
        "option."
    ), "positive"),
]

print("=" * 70)
print("PHASE 12: TEST CASES")
print("=" * 70)

passed = 0
for name, text, expected in TEST_CASES:
    result = predict(text)
    status = "N/A (handled gracefully)" if expected is None else (
        "PASS" if result == expected else "FAIL"
    )
    if status != "FAIL":
        passed += 1
    print(f"\n[{name}]")
    print(f"  Input:    {text if text else '(empty string)'}")
    print(f"  Expected: {expected if expected else 'graceful handling, no crash'}")
    print(f"  Got:      {result}")
    print(f"  Status:   {status}")

print("\n" + "=" * 70)
print(f"Result: {passed}/{len(TEST_CASES)} test cases behaved as expected")
print("=" * 70)
