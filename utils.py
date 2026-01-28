# -*- coding: utf-8 -*-
import math
import networkx as nx

class CostCalculator:
    """
    Proje Dokman Blm 3'teki metrikleri hesaplar.
    """

    def __init__(self, G, w_delay=0.33, w_reliability=0.33, w_resource=0.33):
        self.G = G
        # Ağırlıkları normalize et (Toplamı 1 olacak şekilde)
        total_w = w_delay + w_reliability + w_resource
        if total_w == 0:
            self.w_delay = 0
            self.w_rel = 0
            self.w_res = 0
        else:
            self.w_delay = w_delay / total_w
            self.w_rel = w_reliability / total_w
            self.w_res = w_resource / total_w

    def calculate_path_metrics(self, path):
        """
        Bir yolun Gecikme, Güvenilirlik ve Kaynak Maliyetini hesaplar.
        Dönüş: (total_delay, reliability_cost, resource_cost)
        """
        # Yol boşsa veya geçersizse sonsuz döndür
        if not path or len(path) < 2:
             return float('inf'), float('inf'), float('inf')

        total_delay = 0
        reliability_cost = 0
        resource_cost = 0

        # --- 1. Düğüm (Node) Maliyetleri ---
        # PDF Madde 3.1'e göre: Kaynak (S) ve Hedef (D) işlem süreleri hariç tutulur.
        # path[0] -> Kaynak, path[-1] -> Hedef
        
        for i, node_id in enumerate(path):
            node_data = self.G.nodes[node_id]

            # Gecikme: Sadece ara düğümler için ekle
            if i != 0 and i != len(path) - 1:
                total_delay += node_data.get('s_ms', 0)

            # Güvenilirlik: Tüm düğümler (S ve D dahil) ayakta olmalı
            r_node = node_data.get('r_node', 0.99)
            if r_node > 0:
                reliability_cost += -math.log(r_node)
            else:
                reliability_cost += 100 # Ceza puanı

        # --- 2. Bağlantı (Link) Maliyetleri ---
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]

            # Eğer haritada böyle bir bağ yoksa yol geçersizdir
            if not self.G.has_edge(u, v):
                return float('inf'), float('inf'), float('inf')

            edge_data = self.G.edges[u, v]

            # Link Gecikmesi [cite: 42]
            total_delay += edge_data.get('delay_ms', 0)

            # Link Güvenilirliği [cite: 49] (Logaritmik toplama dönüştürüldü)
            r_link = edge_data.get('r_link', 0.99)
            if r_link > 0:
                reliability_cost += -math.log(r_link)
            else:
                reliability_cost += 100

            # Kaynak Maliyeti (1 Gbps / Bandwidth) [cite: 57]
            bw = edge_data.get('capacity_mbps', 100)
            if bw > 0:
                resource_cost += (1000.0 / bw)
            else:
                resource_cost += 1000 # Bant genişliği 0 ise büyük ceza

        return total_delay, reliability_cost, resource_cost

    def calculate_total_fitness(self, path):
        """
        Ağırlıklı Toplam Yöntemi ile tek bir skor üretir.
        Daha DÜŞÜK puan = Daha İYİ yol.
        """
        d, r, res = self.calculate_path_metrics(path)

        # Eğer yol geçersizse (sonsuz döndüyse)
        if d == float('inf'):
            return float('inf')

        # Formül: W_delay * Delay + W_rel * Rel_Cost + W_res * Res_Cost [cite: 66-68]
        total_cost = (self.w_delay * d) + \
                     (self.w_rel * r) + \
                     (self.w_res * res)

        return total_cost