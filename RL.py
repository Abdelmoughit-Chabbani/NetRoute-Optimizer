# -*- coding: utf-8 -*-
import random
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import pandas as pd
import traceback

from network_generator import NetworkGenerator
from utils import CostCalculator


class QLearningAlgorithm:
    def __init__(
            self,
            graph,
            alpha=0.1,
            gamma=0.9,
            epsilon=0.2,
            episodes=250,
            w_delay=0.33,
            w_reliability=0.33,
            w_resource=0.33,
    ):
        self.G = graph
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.q_table = {}
        # Maliyet hesaplayıcıyı başlat
        self.cost_calc = CostCalculator(
            self.G,
            w_delay=w_delay,
            w_reliability=w_reliability,
            w_resource=w_resource
        )

    def get_q(self, s, a):
        return self.q_table.get((s, a), 0.0)

    def set_q(self, s, a, value):
        self.q_table[(s, a)] = value

    def run(self, S, D, demand_bw=0):
        print(f"Q-Learning eğitiliyor ({S} -> {D}, Talep: {demand_bw} Mbps)...")

        for ep in range(self.episodes):
            current_node = S
            path = [S]

            while current_node != D and len(path) < 100:
                # Bant genişliği kısıtını sağlayan komşuları bul
                neighbors = [n for n in self.G.neighbors(current_node)
                             if self.G.edges[current_node, n].get('capacity_mbps', 0) >= demand_bw]

                if not neighbors: break

                # Epsilon-Greedy seçimi
                if random.random() < self.epsilon:
                    next_node = random.choice(neighbors)
                else:
                    qs = [self.get_q(current_node, n) for n in neighbors]
                    max_q = max(qs)
                    best_neighbors = [neighbors[i] for i, q in enumerate(qs) if q == max_q]
                    next_node = random.choice(best_neighbors)

                # Ödül mekanizması
                if next_node == D:
                    fitness = self.cost_calc.calculate_total_fitness(path + [next_node])
                    reward = 1000.0 / (fitness + 1e-6)
                else:
                    reward = 0

                # Bellman Denklemi güncellemesi
                next_neighbors = [nn for nn in self.G.neighbors(next_node)
                                  if self.G.edges[next_node, nn].get('capacity_mbps', 0) >= demand_bw]
                max_next_q = max([self.get_q(next_node, nn) for nn in next_neighbors]) if next_neighbors else 0

                old_q = self.get_q(current_node, next_node)
                new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
                self.set_q(current_node, next_node, new_q)

                current_node = next_node
                path.append(current_node)

        return self.extract_best_path(S, D, demand_bw)

    def extract_best_path(self, S, D, demand_bw):
        path = [S]
        curr = S
        visited = {S}
        while curr != D:
            neighbors = [n for n in self.G.neighbors(curr)
                         if self.G.edges[curr, n].get('capacity_mbps', 0) >= demand_bw]
            unvisited = [n for n in neighbors if n not in visited]
            if not unvisited: break

            qs = [self.get_q(curr, n) for n in unvisited]
            curr = unvisited[np.argmax(qs)]
            path.append(curr)
            visited.add(curr)
            if len(path) > 100: break
        return path, self.cost_calc.calculate_total_fitness(path)

    def plot_full_network_with_path(self, path):
        pos = nx.spring_layout(self.G, seed=42)
        plt.figure(figsize=(12, 12))
        nx.draw(self.G, pos, node_size=15, node_color="lightgray", edge_color="gray", alpha=0.2)

        if path and len(path) > 1:
            edges = list(zip(path, path[1:]))
            nx.draw_networkx_edges(self.G, pos, edgelist=edges, edge_color="red", width=3)
            nx.draw_networkx_nodes(self.G, pos, nodelist=path, node_color="red", node_size=100)
            labels = {n: str(n) for n in path}
            nx.draw_networkx_labels(self.G, pos, labels, font_size=8, font_color="black")

        plt.title(f"Q-Learning En Iyi Yol (Source: {path[0]} -> Dest: {path[-1]})")
        plt.axis("off")
        plt.show()


def build_graph_from_excel(node_file, edge_file):
    G = nx.Graph()

    # Node okuma
    df_nodes = pd.read_excel(node_file)
    for _, row in df_nodes.iterrows():
        G.add_node(int(row.iloc[0]), s_ms=row.iloc[1], r_node=row.iloc[2])

    # Edge okuma
    df_edges = pd.read_excel(edge_file)
    for _, row in df_edges.iterrows():
        G.add_edge(int(row.iloc[0]), int(row.iloc[1]),
                   delay_ms=row.iloc[2],
                   capacity_mbps=row.iloc[3],
                   r_link=row.iloc[4])
    return G


if __name__ == "__main__":
    # DOSYA YOLLARI
    node_file = "data/NodeData.xlsx"
    edge_file = "data/EdgeData.xlsx"
    demand_file = "data/DemandData.xlsx"

    if os.path.exists(node_file) and os.path.exists(edge_file):
        try:
            print("Ağ yapısı Excel'den yükleniyor...")
            G = build_graph_from_excel(node_file, edge_file)

            print("Trafik talepleri yükleniyor...")
            df_demand = pd.read_excel(demand_file)

            # Demand okuma
            test_case = df_demand.iloc[0]
            S = int(test_case.iloc[0])
            D = int(test_case.iloc[1])
            B = int(test_case.iloc[2])
            B = 0

            print(f"\n--- TEST SENARYOSU: {S} -> {D} ({B} Mbps) ---")

            rl = QLearningAlgorithm(G, episodes=250)
            best_path, best_cost = rl.run(S, D, demand_bw=B)

            # Sonuçları hesapla ve yazdır
            delay, rel_cost, res_cost = rl.cost_calc.calculate_path_metrics(best_path)
            # Güvenilirlik maliyetini (logaritmik) gerçek orana çevir
            total_reliability = math.exp(-rel_cost) if rel_cost != float('inf') else 0

            print("\n" + "=" * 30)
            print("      RL OPTİMİZASYON SONUÇLARI")
            print("=" * 30)
            print(f"En İyi Yol     : {best_path}")
            print(f"Toplam Maliyet : {best_cost:.4f}")
            print(f"Toplam Gecikme : {delay:.2f} ms")
            print(f"Güvenilirlik   : %{total_reliability * 100:.2f}")
            print("=" * 30)

            rl.plot_full_network_with_path(best_path)

        except Exception as e:
            print(f"\nHATA OLUŞTU!\n{e}")
            traceback.print_exc()
    else:
        print("Hata: Excel dosyaları 'data/' klasöründe bulunamadı!")