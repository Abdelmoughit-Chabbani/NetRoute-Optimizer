import random
import math
import networkx as nx
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from utils import CostCalculator
from network_generator import NetworkGenerator

class AntColonyOptimization:
    def __init__(
     self,
     graph,
     num_ants=75,   #Her turda yola çıkan karınca sayısı
     generations=100,    #Toplam iterasyon sayısı
     alpha=1.0,  #Feromon Kuvvetlendirme Oranı: Düğümler arasındaki feromon miktarlarının önem derecesini belirler
     beta=1.5,  #Sezgisellik Kuvvetlendirme Oranı: Düğümler arasındaki mesafenin önem derecesini belirler
     evaporation_rate=0.2,  #Feromon Buharlaşma Oranı: Her iterasyon sonunda düğümler arasındaki feromonların hangi oranda buharlaşacağını belirler
     Q=100.0,   #Bırakılacak feromon miktarı için
     w_delay=0.33,
     w_reliability=0.33,
     w_resource=0.33,
    ):
        self.G=graph
        self.num_ants=num_ants
        self.generations=generations
        self.alpha=alpha
        self.beta=beta
        self.evaporation_rate=evaporation_rate
        self.Q=Q

        #Maliyet hesaplayıcıyı başlat
        self.cost_calc=CostCalculator(self.G, w_delay=w_delay, w_reliability=w_reliability, w_resource=w_resource)

        #Başlangıç feromonları
        self.pheromones={}
        initial_pheromone = 1.0
        for u, v in self.G.edges():
            self.pheromones[(u, v)]=initial_pheromone
            self.pheromones[(v, u)]=initial_pheromone
    

    #Karıncanın bir kenarı ne kadar çekici bulduğunu hesaplar. Çekicilik ne kadar yüksekse maliyeti o kadar düşük olur.
    def _get_heuristic(self, u, v):
        if not self.G.has_edge(u, v):    #u ile v arasında yol yoksa
            return 0.0
        
        edge_data=self.G.edges[u, v]
        node_data=self.G.nodes[v]

        #Gecikme (Düğüm İşlem Süresi + Bağlantı Gecikmesi)
        d=edge_data.get('delay_ms', 0) + node_data.get('s_ms',0)

        r_val=edge_data.get('r_link', 0.99)
        r_cost = -math.log(r_val) if r_val > 0 else 100

        bw = edge_data.get('capacity_mbps', 100)
        res_cost = (1000.0 / bw) if bw > 0 else 1000

        edge_cost = (self.cost_calc.w_delay * d) + \
                    (self.cost_calc.w_rel * r_cost) + \
                    (self.cost_calc.w_res * res_cost)
        if edge_cost <= 0:  # 0'a bölme hatasını önlemek için
            return 100.0
        
        return 1.0 / edge_cost
    

    def _select_next_node(self, current_node, visited, required_bandwidth):
        neighbors=list(self.G.neighbors(current_node))

        #Ziyaret edilmemiş ve bw kapasitesi yeterli olan yerleri seç
        valid_neighbors=[]
        for n in neighbors:
            if n not in visited:
                #Kapasite kontrolü
                capacity=self.G.edges[current_node, n].get('capacity_mbps', 0)
                if capacity >= required_bandwidth:
                    valid_neighbors.append(n)
        
        #Eğer gidecek yer kalmadıysa
        if not valid_neighbors:
            return None
        
        #Sıradaki yolun seçilme olasılıklarını hesapla. P = (Feromon^alpha) * (Heuristic^beta)
        probabilities=[]
        denominator=0.0

        for neighbor in valid_neighbors:
            tau=self.pheromones.get((current_node, neighbor), 1.0)  #Feromon miktarı
            eta=self._get_heuristic(current_node, neighbor) #Yolun çekiciliği

            prob_value= (tau ** self.alpha)*(eta**self.beta)
            probabilities.append(prob_value)
            denominator+=prob_value

        #Olasılıklar toplamı 0 ise rastgele seç
        if denominator==0:
            return random.choice(valid_neighbors)
            
        #Olasılıkları toplamı 1 olacak şekilde normalize et
        probabilities = [p / denominator for p in probabilities]

        #Seçim yap
        next_node=random.choices(valid_neighbors, weights=probabilities, k=1)[0] #weights'deki olasılıklara göre rastgele 1 tane (k=1) seçim yap ve içindeki elemanı ver [0]
        return next_node
        
    
    def run(self, S, D, demand_mbps=0):
        #Önceki testten kalan kokuları temizliyoruzki listede sıradaki satır için sıfırdan başlangıç olsun
        initial_pheromone = 1.0
        for u, v in self.G.edges():
             self.pheromones[(u, v)] = initial_pheromone
             self.pheromones[(v, u)] = initial_pheromone

        best_path=None
        best_cost=float('inf')

        #Belirlenen jenerasyon sayısı kadar iterasyon yap
        for gen in range(self.generations):
            all_paths=[]    #Bu turdaki başarılı karıncaların yolları

            #Karıncalar yola çıksın
            for _ in range(self.num_ants):
                path=[S]
                current=S

                #Karınca hedefe varana kadar ya da sıkışana kadar devam et
                while current!=D:
                    visited_Set=set(path)   #Ziyaret edilenleri hızlı kontrol etmek için kümeye çevir.
                    nxt= self._select_next_node(current, visited_Set, demand_mbps)

                    #Eğer karınca çıkmaza girerse ya da kapasite yetmezse elenir.
                    if nxt is None:
                        break

                    path.append(nxt)
                    current=nxt

                #Eğer karınca hedefe ulaştıysa yolun maliyetini hesapla
                if current==D:
                    cost=self.cost_calc.calculate_total_fitness(path)
                    all_paths.append((path, cost))

                    #En iyi çözümü güncelle
                    if cost<best_cost:
                        best_cost=cost
                        best_path=path
                
            #Her turun sonunda feromonlar biraz buharlaşır
            for key in  self.pheromones:
                self.pheromones[key]*=(1.0-self.evaporation_rate)

            
            #Feromon güncelleme
            for path, cost in all_paths:
                #Maliyet ne kadar düşükse o kadar çok feromon bırakır
                pheromone_trail=self.Q/cost if cost>0 else self.Q

                for i in range(len(path)-1):
                    u, v=path[i], path[i+1]
                    self.pheromones[(u, v)] += pheromone_trail
                    self.pheromones[(v, u)] += pheromone_trail
        
        return best_path, best_cost
    

    #Listeden S,D,B okuyarak sırasıyla en kısa yolu bulma
    def run_from_file(self, filename):
        #Dosya var mı kontrol et
        if not os.path.exists(filename):
            print(f"\nHATA! '{filename}' dosyası bulunamadı!")
            return

        print(f"\n--- {filename} dosyası sırayla hesaplanıyor... ---\n")
        # Başlıkları yazdır
        print(f"{'S->D':<10} {'Talep':<8} {'Maliyet':<8} {'Gecikme':<9} {'Güvenilirlik':<8} {'Süre':<7} {'Yol'}")
        print("-" * 100)

        try:
            #Exceli oku
            df = pd.read_excel(filename)
            
            #Her satırı tek tek dön
            for index, row in df.iterrows():
                try:
                    #Sütun isimleri
                    s_node = int(row['src'])
                    d_node = int(row['dst'])
                    demand = int(row['demand_mbps'])
                except KeyError:
                    print(f"\nHATA! Excel sütun isimleri hatalı! 'src', 'dst', 'demand_mbps' olmalı.")
                    return

                #Algoritmayı çalıştır
                start_time = time.time()
                best_path, best_cost = self.run(s_node, d_node, demand_mbps=demand)
                duration = time.time() - start_time

                #Sonuçlandır
                if best_path:
                    cost_str = f"{best_cost:.2f}"
                    
                    #Metrikleri hesapla (Gecikme ve Güvenilirlik için)
                    d, r_cost, res_cost = self.cost_calc.calculate_path_metrics(best_path)
                    
                    #Güvenilirlik maliyetini yüzdeye çevir
                    total_rel = math.exp(-r_cost) 
                    
                    delay_str = f"{d:.2f}ms"
                    rel_str = f"{total_rel:.4f}"
                    path_str = str(best_path)    #Yolu yazıya çevir
                else:
                    cost_str = "-"
                    delay_str = "-"
                    rel_str = "-"
                    path_str = "YOL BULUNAMADI"

                #Sonuçları ekrana yazdır
                print(f"{s_node}->{d_node:<5} {demand:<8} {cost_str:<8} {delay_str:<9} {rel_str:<8} {duration:.4f}  {path_str}")

        except Exception as e:
            print(f"Hata oluştu: {e}")

    #Grafik Çizimi
    def plot_full_network_with_path(self, path):
        if self.G is None: return

        pos = nx.spring_layout(self.G, seed=42)
        plt.figure(figsize=(11, 11))
        
        # Tüm ağı çiz
        nx.draw(
            self.G, pos,
            node_size=20,
            node_color="lightgray",
            edge_color="gray",
            alpha=0.2
        )
        
        # Bulunan yolu çiz
        if path:
            edges = list(zip(path, path[1:]))
            nx.draw_networkx_edges(
                self.G, pos,
                edgelist=edges,
                edge_color="blue",
                width=3
            )
            nx.draw_networkx_nodes(
                self.G, pos,
                nodelist=path,
                node_color="blue",
                node_size=140
            )
            labels = {n: str(n) for n in path}
            nx.draw_networkx_labels(self.G, pos, labels, font_size=9, font_color="white")

        plt.title(f"ACO – Full Network with Best Path (Length: {len(path) if path else 0})")
        plt.axis("off")
        plt.show()

    def plot_only_path(self, path):
        plt.figure(figsize=(6, 3))
        plt.plot(range(len(path)), path, marker="o", color='b') # GA'da default renk, burada mavi
        plt.xlabel("Path Step")
        plt.ylabel("Node ID")
        plt.title("Best Path Sequence")
        plt.grid(True)
        plt.show()

    def plot_path_metrics(self, delay, rel_cost, res_cost):
        labels = ["Delay", "Reliability Cost", "Resource Cost"]
        values = [delay, rel_cost, res_cost]

        plt.figure(figsize=(6, 4))
        plt.bar(labels, values, color=['#d9534f', '#5cb85c', '#f0ad4e'])
        plt.title("Cost Components of Best Path")
        plt.grid(axis="y", alpha=0.3)
        plt.show()



# TEST BLOĞU
if __name__ == "__main__":
    # 1. Ağı Oluştur
    gen = NetworkGenerator()
    gen.create_topology() 
    G = gen.G

    print(f"Ağ Oluşturuldu: {G.number_of_nodes()} Düğüm, {G.number_of_edges()} Bağlantı")

    # 2. ACO Algoritmasını Başlat
    aco = AntColonyOptimization(G)

    #3. Kullanıcıya Seçenek Sun
    print("\n[1] Manuel Test (Kaynak ve Hedefi ben gireceğim)")
    print("[2] Excel'den Toplu Test (DemandData.xlsx okunacak)")
    secim = input("Seçiminiz (1 veya 2): ")

    if secim == '2':
        #Excel dosyasının yolunu belirle. Dosya "data" klasörünün içinde.
        excel_dosya_adi = os.path.join("data", "DemandData.xlsx")
        aco.run_from_file(excel_dosya_adi)
    
    else:
        #Manuel mod
        try:
            S = int(input("\nKaynak düğümü gir (0-249): "))
            D = int(input("Hedef düğümü gir (0-249): "))
            DEMAND = int(input("Talep Edilen Bant Genişliği (Mbps): "))
        except ValueError:
            print("Hatalı giriş! Varsayılan: 0 -> 249, 100 Mbps")
            S, D, DEMAND = 0, 249, 100

        print(f"\nHesaplanıyor: {S} -> {D} (Talep: {DEMAND} Mbps)...")

        start_time = time.time()
        best_path, best_cost = aco.run(S, D, demand_mbps=DEMAND)
        duration = time.time() - start_time

        if best_path:
            #Raporlama
            d, r_cost, res_cost = aco.cost_calc.calculate_path_metrics(best_path)
            total_rel = math.exp(-r_cost)

            print("\n[ACO SONUÇLARI]")
            print(f"En iyi yol       : {best_path}")
            print(f"Toplam Gecikme   : {d:.2f} ms")
            print(f"Güvenilirlik     : {total_rel:.6f}")
            print(f"Maliyet (Cost)   : {best_cost:.4f}")
            print(f"Süre             : {duration:.4f} sn")
            
            # Grafikleri Çiz
            aco.plot_full_network_with_path(best_path)
            aco.plot_only_path(best_path)
            aco.plot_path_metrics(d, r_cost, res_cost)
        else:
            print(f"\nHATA! {S} ile {D} arasında {DEMAND} Mbps kapasiteli uygun yol bulunamadı!")