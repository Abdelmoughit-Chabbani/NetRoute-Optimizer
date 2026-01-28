# -*- coding: utf-8 -*-

# =================================================
# GEREKLİ KÜTÜPHANELER
# =================================================
import random
import math
import os
import pandas as pd
import networkx as nx

from network_generator import NetworkGenerator
from utils import CostCalculator


# =================================================
# GENETIC ALGORITHM (GA)
# =================================================
class GeneticAlgorithm:
    def __init__(
        self,
        graph,
        population_size=40,
        crossover_rate=0.8,
        mutation_rate=0.1,
        generations=100,
        elite_ratio=0.15,
        w_delay=0.33,
        w_reliability=0.33,
        w_resource=0.33,
    ):
        self.G = graph
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generations = generations

        self.elite_size = max(1, int(population_size * elite_ratio))

        self.cost_calc = CostCalculator(
            self.G,
            w_delay=w_delay,
            w_reliability=w_reliability,
            w_resource=w_resource
        )

    def check_bandwidth(self, path, demand):
        for i in range(len(path) - 1):
            if not self.G.has_edge(path[i], path[i + 1]):
                return False
            bw = self.G.edges[path[i], path[i + 1]].get("capacity_mbps", 0)
            if bw < demand:
                return False
        return True

    def random_valid_path(self, S, D, demand):
        try:
            paths = list(nx.all_shortest_paths(self.G, S, D))
            random.shuffle(paths)
            for path in paths:
                if self.check_bandwidth(path, demand):
                    return path
        except:
            pass
        return None

    def initialize_population(self, S, D, demand):
        pop = []
        attempts = 0
        while len(pop) < self.population_size and attempts < self.population_size * 20:
            p = self.random_valid_path(S, D, demand)
            attempts += 1
            if p:
                pop.append(p)
        return pop

    def fitness(self, path, demand):
        if not path or not self.check_bandwidth(path, demand):
            return float("inf")
        return self.cost_calc.calculate_total_fitness(path)

    def select_elite(self, population, demand):
        population.sort(key=lambda p: self.fitness(p, demand))
        return population[:self.elite_size]

    def crossover(self, p1, p2):
        common = set(p1[1:-1]) & set(p2[1:-1])
        if not common:
            return p1[:]
        cp = random.choice(list(common))
        return p1[:p1.index(cp)] + p2[p2.index(cp):]

    def mutate(self, path):
        if len(path) < 4:
            return path
        idx = random.randint(1, len(path) - 2)
        neighbors = list(self.G.neighbors(path[idx]))
        if neighbors:
            path[idx] = random.choice(neighbors)
        return path

    def run(self, S, D, demand):
        population = self.initialize_population(S, D, demand)
        if not population:
            return None, float("inf")

        best = min(population, key=lambda p: self.fitness(p, demand))

        for gen in range(1, self.generations + 1):
            new_population = self.select_elite(population, demand)

            while len(new_population) < self.population_size:
                p1, p2 = random.sample(population, 2)
                child = p1[:]

                if random.random() < self.crossover_rate:
                    child = self.crossover(p1, p2)

                if random.random() < self.mutation_rate:
                    child = self.mutate(child)

                if child[0] == S and child[-1] == D:
                    new_population.append(child)

            population = new_population

            current_best = min(population, key=lambda p: self.fitness(p, demand))
            if self.fitness(current_best, demand) < self.fitness(best, demand):
                best = current_best

            if gen == 1 or gen == self.generations:
                print(f"[GA] Nesil {gen}/{self.generations} - En iyi maliyet: {self.fitness(best, demand):.4f}")

        return best, self.fitness(best, demand)


# =================================================
# MAIN
# =================================================
if __name__ == "__main__":

    random.seed()

    # AĞ OLUŞTUR
    gen = NetworkGenerator()
    gen.create_topology()
    G = gen.G

    # -------------------------------------------------
    # AĞ METRİKLERİ
    # -------------------------------------------------
    edge_bw = nx.get_edge_attributes(G, "capacity_mbps")
    edge_delay = nx.get_edge_attributes(G, "delay_ms")

    avg_bw = sum(edge_bw.values()) / len(edge_bw) if edge_bw else 0
    avg_delay = sum(edge_delay.values()) / len(edge_delay) if edge_delay else 0

    # =================================================
    # DEMAND DOSYASI
    # =================================================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DEMAND_PATH = os.path.join(BASE_DIR, "..", "data", "DemandData.xlsx")

    df = pd.read_excel(DEMAND_PATH)
    demands = df.to_dict(orient="records")

    ga = GeneticAlgorithm(G)

    # GA ORTALAMA GÜVENİRLİK İÇİN
    ga_reliabilities = []

    # =================================================
    # TÜM DEMAND SATIRLARI
    # =================================================
    for i, d in enumerate(demands, start=1):
        S = int(d["src"])
        D = int(d["dst"])
        DEMAND = int(d["demand_mbps"])

        print("\n===================================")
        print(f"DEMAND {i}: {S} -> {D} | Minimum BW: {DEMAND} Mbps")

        best_path, best_cost = ga.run(S, D, DEMAND)

        if best_path:
            delay, r_cost, res_cost = ga.cost_calc.calculate_path_metrics(best_path)
            total_rel = math.exp(-r_cost)
            ga_reliabilities.append(total_rel)

            used_bw = min(
                G.edges[best_path[j], best_path[j + 1]]["capacity_mbps"]
                for j in range(len(best_path) - 1)
            )

            print("[GA SONUÇLARI]")
            print(f"En iyi yol (path)        : {best_path}")
            print(f"Kullanılan Min BW        : {used_bw} Mbps")
            print(f"Toplam Gecikme           : {delay:.2f} ms")
            print(f"Toplam Güvenilirlik      : {total_rel:.4f}")
            print(f"Güvenilirlik Maliyeti    : {r_cost:.4f}")
            print(f"Kaynak Maliyeti          : {res_cost:.4f}")
            print(f"Toplam Maliyet (Cost)    : {best_cost:.4f}")
        else:
            print("Uygun yol bulunamadı (Bandwidth kısıtı).")

    # =================================================
    # AĞ BİLGİLERİ
    # =================================================
    avg_ga_rel_percent = (sum(ga_reliabilities) / len(ga_reliabilities)) * 100 if ga_reliabilities else 0

    print("\nAĞ BİLGİLERİ:")
    print(f"- Düğüm sayısı    : {G.number_of_nodes()}")
    print(f"- Bağlantı sayısı : {G.number_of_edges()}")
    print(f"- Graf yoğunluğu  : {nx.density(G):.4f}")
    print(f"- Bağlı graf mı?  : {'Evet' if nx.is_connected(G) else 'Hayır'}")
    print(f"- Ortalama Bant Genişliği : {avg_bw:.2f} Mbps")
    print(f"- Ortalama Gecikme        : {avg_delay:.2f} ms")
    print(f"- Ortalama Güvenilirlik   : %{avg_ga_rel_percent:.2f}")