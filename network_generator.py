import networkx as nx
import random
import pandas as pd
import os

class NetworkGenerator:
    def __init__(self, num_nodes=250, link_prob=0.2, seed=42):
        self.num_nodes = num_nodes
        self.link_prob = link_prob
        self.seed = seed
        self.G = None
        
        # Tekrarlanabilir sonuclar icin seed
        random.seed(self.seed)


    def visualize(self):
        """
        Ağı ekrana çizdirir.
        """
        import matplotlib.pyplot as plt
        
        if self.G is None:
            print("Görselleştirilecek bir ağ yok!")
            return

        print("Ağ görselleştiriliyor, lütfen bekleyin...")
        
        # Çizim alanı boyutu
        plt.figure(figsize=(12, 10))
        
        # Düğümlerin yerleşimi (Spring Layout: Yaylanma mantığıyla dağıtır)
        pos = nx.spring_layout(self.G, seed=42) 
        
        # Çizim komutları
        nx.draw_networkx_nodes(self.G, pos, node_size=30, node_color='skyblue')
        nx.draw_networkx_edges(self.G, pos, alpha=0.3, edge_color='gray')
        
        # Başlık ve Ayarlar
        plt.title(f"Ağ Topolojisi (N={self.G.number_of_nodes()}, E={self.G.number_of_edges()})")
        plt.axis('off') # Eksen çizgilerini kapat
        
        # Ekrana bas
        plt.show()

    def create_topology(self):
        """
        MOD 1: Rastgele Ag Olusturma
        """
        # Turkce karakter kullanmiyoruz: s, i, g, u
        print(f"Rastgele topoloji olusturuluyor (N={self.num_nodes})...")
        
        while True:
            self.G = nx.erdos_renyi_graph(n=self.num_nodes, p=self.link_prob, seed=self.seed)
            if nx.is_connected(self.G):
                break
            self.seed += 1 

        # Ozellikleri Ata
        for node in self.G.nodes():
            self.G.nodes[node]['s_ms'] = round(random.uniform(0.5, 2.0), 3)
            self.G.nodes[node]['r_node'] = round(random.uniform(0.95, 0.999), 5)

        for u, v in self.G.edges():
            self.G.edges[u, v]['capacity_mbps'] = random.randint(100, 1000)
            self.G.edges[u, v]['delay_ms'] = round(random.uniform(3, 15), 3)
            self.G.edges[u, v]['r_link'] = round(random.uniform(0.95, 0.999), 5)
        
        print("Rastgele ag hazir.")

    def load_from_csv(self, node_file, edge_file):
        """
        MOD 2: Hocanin Verdigi CSV'den Ag Yukleme
        """
        print(f"Dosyadan yukleniyor: {node_file}")
        
        try:
            df_nodes = pd.read_csv(node_file)
            df_edges = pd.read_csv(edge_file)
            
            self.G = nx.Graph()
            
            # 1. Dugumleri Ekle
            for _, row in df_nodes.iterrows():
                node_id = int(row['node_id'])
                self.G.add_node(node_id, 
                                s_ms=float(row['s_ms']), 
                                r_node=float(row['r_node']))
                
            # 2. Baglantilari Ekle
            for _, row in df_edges.iterrows():
                u = int(row['src'])
                v = int(row['dst'])
                self.G.add_edge(u, v, 
                                capacity_mbps=int(row['capacity_mbps']),
                                delay_ms=float(row['delay_ms']),
                                r_link=float(row['r_link']))
                                
            print(f"Dosyadan ag basariyla yuklendi! (Dugum: {self.G.number_of_nodes()}, Bag: {self.G.number_of_edges()})")
            
        except Exception as e:
            print(f"HATA: Dosyalar okunamadi! {e}")

    def load_traffic_demands(self, demand_file):
        """
        Talep dosyasini okur
        """
        print(f"Trafik talepleri yukleniyor: {demand_file}")
        demands = []
        try:
            df_demands = pd.read_csv(demand_file)
            for _, row in df_demands.iterrows():
                demands.append({
                    'source': int(row['src']),
                    'destination': int(row['dst']),
                    'bandwidth': int(row['demand_mbps'])
                })
            print(f"{len(demands)} adet test senaryosu yuklendi.")
            return demands
        except Exception as e:
            print(f"HATA: Talep dosyasi okunamadi! {e}")
            return []

    def save_to_csv(self, path_prefix=""):
        """
        Olusturulan agi CSV olarak kaydeder.
        Eksik olan fonksiyon buydu.
        """
        if self.G is None:
            print("Kaydedilecek bir ag yok!")
            return

        # 1. Node Data
        node_data = []
        for n, data in self.G.nodes(data=True):
            node_data.append({
                'node_id': n,
                's_ms': data.get('s_ms', 0),
                'r_node': data.get('r_node', 0)
            })
        
        # Dosya yollarini birlestir (path_prefix + dosya adi)
        node_path = os.path.join(path_prefix, 'My_NodeData.csv')
        pd.DataFrame(node_data).to_csv(node_path, index=False)
        print(f"--> My_NodeData.csv olusturuldu: {node_path}")

        # 2. Edge Data
        edge_data = []
        for u, v, data in self.G.edges(data=True):
            edge_data.append({
                'src': u,
                'dst': v,
                'capacity_mbps': data.get('capacity_mbps', 0),
                'delay_ms': data.get('delay_ms', 0),
                'r_link': data.get('r_link', 0)
            })
            
        edge_path = os.path.join(path_prefix, 'My_EdgeData.csv')
        pd.DataFrame(edge_data).to_csv(edge_path, index=False)
        print(f"--> My_EdgeData.csv olusturuldu: {edge_path}")
        
        # 3. Demand Data (Test Senaryolari)
        demand_data = []
        possible_nodes = list(self.G.nodes())
        if len(possible_nodes) > 2:
            for _ in range(20):
                s, d = random.sample(possible_nodes, 2)
                bw_req = random.randint(10, 500)
                demand_data.append({
                    'src': s,
                    'dst': d,
                    'demand_mbps': bw_req
                })
            
            demand_path = os.path.join(path_prefix, 'My_DemandData.csv')
            pd.DataFrame(demand_data).to_csv(demand_path, index=False)
            print(f"--> My_DemandData.csv olusturuldu: {demand_path}")

    def get_graph(self):
        return self.G

# --- TEST KISMI (MAIN) ---
if __name__ == "__main__":
    
    # 1. Klasor yollarini dinamik bul
    current_dir = os.path.dirname(os.path.abspath(__file__)) # src klasoru
    base_dir = os.path.dirname(current_dir) # Proje ana klasoru
    data_dir = os.path.join(base_dir, 'data') # data klasoru
    
    # Data klasoru yoksa olustur
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Uyari: '{data_dir}' klasoru olusturuldu.")

    gen = NetworkGenerator()
    
    # Hocanin dosya isimleri
    node_file = "NodeData.xlsx - in.csv"
    edge_file = "EdgeData.xlsx - in.csv"
    
    node_path = os.path.join(data_dir, node_file)
    edge_path = os.path.join(data_dir, edge_file)
    
    # Dosyalar var mi kontrol et
    if os.path.exists(node_path) and os.path.exists(edge_path):
        print(f"Hocanin dosyalari bulundu: {data_dir}")
        gen.load_from_csv(node_path, edge_path)
    else:
        print("Hocanin dosyalari bulunamadi, rastgele olusturuluyor...")
        gen.create_topology()
        
        # Rastgele olusturulani data klasorune kaydet
        # Buradaki save_to_csv hatasi artik cozuldu
        gen.save_to_csv(path_prefix=data_dir)

        gen.visualize()