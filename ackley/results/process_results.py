import json
import numpy as np

with open("ackley_es_results_n_mutation.json") as fp:
    data = json.load(fp)
    min_errors = np.array([d["min error function"] for d in data])
    min_dists = np.array([d["min error solution"] for d in data])

    avg_min_error = min_errors.mean()
    std_min_error = min_errors.std()
    avg_min_dists = min_dists.mean()
    std_min_dists = min_dists.std()

    print("Average min error: {}\nStd dev of min error: {}".format(avg_min_error, std_min_error))
    print("Minimal error: {}".format(min_errors.min()))
    print("###########")
    print("Average min distance: {}\nStd dev of min distance: {}".format(avg_min_dists, std_min_dists))
    print("Minimal distance: {}".format(min_dists.min()))