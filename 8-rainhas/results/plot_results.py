import pandas as pd
import matplotlib.pyplot as plt

def plot_results_for_probability_changes():
    df1 = pd.read_csv("base.csv")
    df2 = pd.read_csv("base_pc_100_pm_80.csv")
    df3 = pd.read_csv("base_pc_80_pm_5.csv")

    df_iterations = pd.DataFrame({
        "90%% crossover, 40%% mutação": df1["iterations"],
        "100%% crossover, 80%% mutação": df2["iterations"],
        "80%% crossover, 5%% mutação": df3["iterations"]
    })

    df_avg_fitness = pd.DataFrame({
        "90%% crossover, 40%% mutação": df1["average fitness"],
        "100%% crossover, 80%% mutação": df2["average fitness"],
        "80%% crossover, 5%% mutação": df3["average fitness"]
    })

    df_iterations.boxplot()
    plt.show()
    df_avg_fitness.boxplot()
    plt.show()

def plot_results_for_pop_size_changes():
    df1 = pd.read_csv("base_pc_100_pm_80_pop_20.csv")
    df2 = pd.read_csv("base_pc_100_pm_80_pop_50.csv")
    df3 = pd.read_csv("base_pc_100_pm_80.csv")
    df4 = pd.read_csv("base_pc_100_pm_80_pop_200.csv")

    df_iterations = pd.DataFrame({
        "20 indivíduos": df1["iterations"],
        "50 indivíduos": df2["iterations"],
        "100 indivíduos": df3["iterations"],
        "200 indivíduos": df4["iterations"]
    })

    df_avg_fitness = pd.DataFrame({
        "20 indivíduos": df1["average fitness"],
        "50 indivíduos": df2["average fitness"],
        "100 indivíduos": df3["average fitness"],
        "200 indivíduos": df4["average fitness"]
    })

    df_iterations.boxplot()
    plt.show()
    df_avg_fitness.boxplot()
    plt.show()

def plot_results_for_crossover_changes():
    df1 = pd.read_csv("base_pc_100_pm_80_pop_200.csv")
    df2 = pd.read_csv("pmx_pc_100_pm_80_pop_200.csv")
    df3 = pd.read_csv("edge_pc_100_pm_80_pop_200.csv")
    df4 = pd.read_csv("cyclic_pc_100_pm_80_pop_200.csv")

    df_iterations = pd.DataFrame({
        "Cut and crossfill": df1["iterations"],
        "PMX": df2["iterations"],
        "Edge crossfill": df3["iterations"],
        "Ciclos": df4["iterations"]
    })

    df_avg_fitness = pd.DataFrame({
        "Cut and crossfill": df1["average fitness"],
        "PMX": df2["average fitness"],
        "Edge crossfill": df3["average fitness"],
        "Ciclos": df4["average fitness"]
    })

    df_iterations.boxplot()
    plt.show()
    df_avg_fitness.boxplot()
    plt.show()

def plot_results_for_mutation_changes():
    df1 = pd.read_csv("pmx_pc_100_pm_80_pop_200.csv")
    df2 = pd.read_csv("pmx_insert_pc_100_pm_80_pop_200.csv")
    df3 = pd.read_csv("pmx_inversion_pc_100_pm_80_pop_200.csv")
    df4 = pd.read_csv("pmx_scramble_pc_100_pm_80_pop_200.csv")

    df_iterations = pd.DataFrame({
        "Troca de genes": df1["iterations"],
        "Inserção": df2["iterations"],
        "Inversão": df3["iterations"],
        "Perturbação": df4["iterations"]
    })

    df_avg_fitness = pd.DataFrame({
        "Troca de genes": df1["average fitness"],
        "Inserção": df2["average fitness"],
        "Inversão": df3["average fitness"],
        "Perturbação": df4["average fitness"]
    })

    df_iterations.boxplot()
    plt.show()
    df_avg_fitness.boxplot()
    plt.show()

def plot_results_for_parent_selection_changes():
    df1 = pd.read_csv("pmx_pc_100_pm_80_pop_200.csv")
    df2 = pd.read_csv("pmx_roulette_pc_100_pm_80_pop_200.csv")

    df_iterations = pd.DataFrame({
        "2 melhores de 5": df1["iterations"],
        "Roleta": df2["iterations"]
    })

    df_avg_fitness = pd.DataFrame({
        "2 melhores de 5": df1["average fitness"],
        "Roleta": df2["average fitness"]
    })

    df_iterations.boxplot()
    plt.show()
    df_avg_fitness.boxplot()
    plt.show()

def plot_results_for_survivor_selection_changes():
    df1 = pd.read_csv("pmx_pc_100_pm_80_pop_200.csv")
    df2 = pd.read_csv("pmx_generational_pc_100_pm_80_pop_200.csv")

    df_iterations = pd.DataFrame({
        "Substituição dos piores": df1["iterations"],
        "Geracional": df2["iterations"]
    })

    df_avg_fitness = pd.DataFrame({
        "Substituição dos piores": df1["average fitness"],
        "Geracional": df2["average fitness"]
    })

    df_iterations.boxplot()
    plt.show()
    df_avg_fitness.boxplot()
    plt.show()

if __name__ == "__main__":
    plot_results_for_survivor_selection_changes()