from cProfile import label
import matplotlib.pyplot as plt
import pandas as pd
import ast

results = pd.read_csv("res.csv")

for method,classifier,train_auc,val_auc in  zip(results['NetWork'],results['Classifier'],results['Train Res'],results['Val_Res']):
    plt.plot(ast.literal_eval(train_auc),label="Train")
    plt.plot(ast.literal_eval(val_auc),label="Val")
    if classifier == "default": classifier = "fully connected layers"
    plt.title(method+" + "+classifier)
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("AUC")
    plt.show()
    print(train_auc,val_auc)