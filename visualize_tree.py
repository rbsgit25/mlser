import joblib
from sklearn import tree
import matplotlib.pyplot as plt


MODEL_PATH = "model.joblib"
PNG_OUT = "decision_tree.png"


model = joblib.load(MODEL_PATH)


plt.figure(figsize=(12,8))
tree.plot_tree(model, feature_names=["UserAge","BrowsingTime","PastPurchases","CartAdds"], filled=True, rounded=True)
plt.tight_layout()
plt.savefig(PNG_OUT, dpi=150)
print("Saved", PNG_OUT)