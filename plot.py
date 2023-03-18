import matplotlib.pyplot as plt

from data_processing import load_data, get_datasets
from eval import evaluate

df = load_data()
datasets = get_datasets(df)

steps = [1, 2, 3]
results = {}
for model_name in ["sma", "arima", "var"]:
    results[model_name] = evaluate(model_name, datasets)

for strain in ["alpha", "delta", "omicron"]:
    for model_name, model_results in results.items():
        plt.plot(steps, model_results[model_results["strain"] == strain]["err_test"], label=model_name.upper())
    plt.legend()
    plt.title(strain)
    plt.xticks(steps)
    plt.xlabel("Forecast Week")
    plt.ylabel("Mean Average Percentage Error")
    plt.show()
