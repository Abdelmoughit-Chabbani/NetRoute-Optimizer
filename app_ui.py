import streamlit as st
import networkx as nx
import pandas as pd
import random
import time
import math
import numpy as np
import plotly.graph_objects as go
from network_generator import NetworkGenerator
from ACO import AntColonyOptimization
from GA import GeneticAlgorithm
from RL import QLearningAlgorithm
from utils import CostCalculator
import os

# ------------------------------------------------------------------------------
# SAYFA YAPILANDIRMASI & TEMA (PAGE CONFIGURATION & THEME)
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="OrbitalComm Komuta Merkezi",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# YUKSEK KALITE ASKERI / GLASSMORPHISM CSS
st.markdown("""
<style>
    /* Arka Plan - Derin Uzay */
    .stApp {
        background-color: #000000;
        background-image: radial-gradient(circle at 50% 50%, #111122 0%, #000000 100%);
    }
    
    /* Tipografi */
    h1, h2, h3, h4, .stMarkdown {
        font-family: 'Segoe UI', 'Roboto', Helvetica, Arial, sans-serif;
        color: #e0e0e0;
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.3);
    }
    
    /* Yan Menü (Sidebar) - Glassmorphism */
    section[data-testid="stSidebar"] {
        background-color: rgba(10, 20, 30, 0.6);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Butonlar */
    .stButton>button {
        background: linear-gradient(45deg, #0044cc, #0088ff);
        color: white;
        border: none;
        border-radius: 4px;
        box-shadow: 0 0 15px rgba(0, 136, 255, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 0 25px rgba(0, 136, 255, 0.7);
        transform: scale(1.02);
    }

    /* Kartlar / Metrikler */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        color: #00ffff !important;
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
    }
    div[data-testid="stMetricLabel"] {
        color: #8899aa !important;
    }
    
    /* Girdiler */
    .stNumberInput input, .stSelectbox, .stSlider {
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# OTURUM DURUMU YÖNETİMİ (SESSION STATE)
# ------------------------------------------------------------------------------
if 'G' not in st.session_state:
    st.session_state.G = None
if 'pos_geo' not in st.session_state:
    st.session_state.pos_geo = None # {node: (lat, lon)}
if 'last_path' not in st.session_state:
    st.session_state.last_path = None
if 'last_metrics' not in st.session_state:
    st.session_state.last_metrics = None

# ------------------------------------------------------------------------------
# YARDIMCI FONKSİYONLAR
# ------------------------------------------------------------------------------
def generate_geo_positions(G, seed=42):
    """
    Graf düğümlerini 3D Küre görünümü için sanal Enlem/Boylam koordinatlarına eşler.
    """
    # 1. 2D yerleşimi al (x, y yaklaşık -1..1 arası)
    pos_2d = nx.spring_layout(G, seed=seed, iterations=50)
    
    geo_pos = {}
    for node, (x, y) in pos_2d.items():
        # x -> Boylam [-180, 180]
        # y -> Enlem [-90, 90]
        lon = x * 160 
        lat = y * 80
        geo_pos[node] = (lat, lon)
    return geo_pos

def create_globe_fig(G, pos_geo, path=None, height=600):
    """
    Yüksek Kalite 3D Ortografik Küre (Uydu Görünümü) Oluşturur.
    Animasyon ve Renkli Düğümler içerir.
    """
    # 1. Kenar (Edge) Verilerini Hazırla
    edge_lats = []
    edge_lons = []
    
    for u, v in G.edges():
        lat0, lon0 = pos_geo[u]
        lat1, lon1 = pos_geo[v]
        edge_lats.extend([lat0, lat1, None])
        edge_lons.extend([lon0, lon1, None])
    
    # Temel Ağ
    edge_trace = go.Scattergeo(
        lat=edge_lats,
        lon=edge_lons,
        mode='lines',
        line=dict(width=0.5, color='rgba(255, 255, 255, 0.15)'), # Daha silik beyazımsı
        hoverinfo='none',
        name='Bağlantılar'
    )
    
    # 2. Vurgulanan Yol (Statik Glow)
    path_trace = None
    if path and len(path) > 1:
        p_lats, p_lons = [], []
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            p_lats.extend([pos_geo[u][0], pos_geo[v][0], None])
            p_lons.extend([pos_geo[u][1], pos_geo[v][1], None])
            
        path_trace = go.Scattergeo(
            lat=p_lats,
            lon=p_lons,
            mode='lines',
            line=dict(width=5, color='#00ffcc'), # Neon Turkuaz Glow
            opacity=0.8,
            name='En İyi Rota'
        )

    # 3. Düğümler (Uydular) - Farklı Renkler
    node_lats = []
    node_lons = []
    node_text = []
    node_colors = []
    node_sizes = []
    
    path_set = set(path) if path else set()
    
    # Renk Paleti için
    cmap = ['#FF595E', '#FFCA3A', '#8AC926', '#1982C4', '#6A4C93', '#F15BB5', '#00BBF9', '#00F5D4']
    
    for i, node in enumerate(G.nodes()):
        lat, lon = pos_geo[node]
        node_lats.append(lat)
        node_lons.append(lon)
        
        # Metadata
        d_val = G.nodes[node].get('s_ms', 0)
        node_text.append(f"<b>UYDU-{node}</b><br>Gecikme: {d_val}ms")
        
        if node in path_set:
            node_colors.append('#FFFFFF') # Yoldaki düğümler Beyaz parlasın
            node_sizes.append(12)
        else:
            # Rastgele veya ID'ye bağlı farklı renkler
            color_idx = (node * 13) % len(cmap)
            node_colors.append(cmap[color_idx])
            node_sizes.append(7)

    node_trace = go.Scattergeo(
        lat=node_lats,
        lon=node_lons,
        mode='markers',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1, color='black'),
            symbol='circle',
            opacity=0.9
        ),
        text=node_text,
        hoverinfo='text',
        name='Uydular'
    )

    data = [edge_trace, node_trace]
    if path_trace: data.append(path_trace)

    # 4. ANİMASYON (Gezgin Parçacık)
    frames = []
    if path and len(path) > 1:
        # Animasyon için bir "Gezgin" trace ekle (başlangıçta boş veya ilk noktada)
        traveler_trace = go.Scattergeo(
            lat=[pos_geo[path[0]][0]],
            lon=[pos_geo[path[0]][1]],
            mode='markers',
            marker=dict(size=20, color='yellow', symbol='star', line=dict(width=2, color='orange')),
            name='Gezgin',
            showlegend=False
        )
        data.append(traveler_trace)
        
        # Kareleri (Frames) oluştur
        traveler_idx = len(data) - 1
        for k in range(len(path)):
            curr_node = path[k]
            frames.append(go.Frame(
                data=[go.Scattergeo(
                    lat=[pos_geo[curr_node][0]],
                    lon=[pos_geo[curr_node][1]],
                    mode='markers',
                    marker=dict(size=25, color='yellow', symbol='star')
                )],
                name=f'frame{k}',
                traces=[traveler_idx] 
            ))

    # Düzen Ayarları
    layout = go.Layout(
        title="",
        showlegend=False,
        geo=dict(
            projection_type="orthographic",
            showland=True,
            showocean=True,
            showcountries=False,
            # Karanlık Dünya
            landcolor="rgb(15, 15, 20)",
            oceancolor="rgb(5, 5, 10)", 
            lakecolor="rgb(5, 5, 10)",
            coastlinecolor="rgba(255, 255, 255, 0.1)",
            bgcolor='rgba(0,0,0,0)',
            lonaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            lataxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=height,
        # Animasyon Butonları
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            x=0.1, y=0.1,
            buttons=[dict(label="▶",
                          method="animate",
                          args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True)])]
        )] if frames else []
    )
    
    fig = go.Figure(data=data, layout=layout, frames=frames)
    return fig

def display_network_metrics(G):
    """
    Ağ metriklerini ve istatistiklerini detaylı gösterir.
    """
    if G is None: return

    # Hesaplamalar
    n = G.number_of_nodes()
    e = G.number_of_edges()
    density = nx.density(G)
    
    # Ortalama Değerler
    total_bw = 0
    total_delay = 0
    total_rel = 0
    edge_count = 0
    
    if e > 0:
        for u, v, d in G.edges(data=True):
            total_bw += d.get('capacity_mbps', 0)
            total_delay += d.get('delay_ms', 0)
            total_rel += d.get('r_link', 0)
        
        avg_bw = total_bw / e
        avg_delay = total_delay / e
        avg_rel = total_rel / e
    else:
        avg_bw = 0
        avg_delay = 0
        avg_rel = 0

    st.markdown("""
    <div style="background: rgba(0, 20, 40, 0.5); padding: 15px; border-radius: 8px; border: 1px solid rgba(0, 255, 255, 0.2);">
        <h4 style="margin-top:0; border-bottom:1px solid rgba(255,255,255,0.1); padding-bottom:10px; color:#00ffff;">🌐 Ağ İstatistikleri</h4>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Düğüm Sayısı:** {n}")
        st.write(f"**Kenar Sayısı:** {e}")
        st.write(f"**Yoğunluk:** {density:.4f}")
    with c2:
        st.write(f"**Ort. Bant Gen.:** {avg_bw:.0f} Mbps")
        st.write(f"**Ort. Gecikme:** {avg_delay:.2f} ms")
        st.write(f"**Ort. Güven.:** {avg_rel:.4f}")
        
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# YAN MENÜ KONTROLLERİ (SIDEBAR)
# ------------------------------------------------------------------------------
st.sidebar.markdown("## 📡 AĞ YAPILANDIRMASI")
st.sidebar.markdown("---")

st.sidebar.markdown("## 📡 AĞ YAPILANDIRMASI")
st.sidebar.markdown("---")

# 1) Algoritma/Topoloji Kaynagi Secimi
topo_choice = st.sidebar.selectbox("Topoloji Kaynağı", ["Rastgele Ağ (Erdos-Renyi)", "Özel Algoritma (CSV'den Yükle)"])

if topo_choice == "Rastgele Ağ (Erdos-Renyi)":
    num_nodes = st.sidebar.slider("Uydu Sayısı", 50, 500, 200)
    link_prob = st.sidebar.slider("Bağlantı Yoğunluğu", 0.01, 1.0, 0.08)
    seed_val = st.sidebar.number_input("Seed Anahtarı", value=42)
else:
    st.sidebar.info("📂 'data/' klasöründeki NodeData ve EdgeData dosyaları kullanılacak.")
    st.sidebar.caption("Lütfen data klasöründe 'NodeData.xlsx - in.csv' ve 'EdgeData.xlsx - in.csv' dosyalarının olduğundan emin olun.")
    seed_val = 42

if st.sidebar.button("TAKIMYILDIZI YÖRÜNGEYE YERLEŞTİR", type="primary"):
    with st.spinner("Yörünge Parametreleri Başlatılıyor..."):
        # Parametreleri guvenli sekilde al
        n_nodes = num_nodes if 'num_nodes' in locals() else 250
        l_prob = link_prob if 'link_prob' in locals() else 0.08
        
        gen = NetworkGenerator(num_nodes=n_nodes, link_prob=l_prob, seed=int(seed_val))
        
        if topo_choice == "Rastgele Ağ (Erdos-Renyi)":
            gen.create_topology()
        else:
            # Custom CSV Loading Logic - Backend degistirmeden
            # Dosya yollarini UI tarafinda buluyoruz
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(base_dir, 'data')
            node_path = os.path.join(data_dir, "NodeData.xlsx - in.csv")
            edge_path = os.path.join(data_dir, "EdgeData.xlsx - in.csv")
            
            if os.path.exists(node_path) and os.path.exists(edge_path):
                gen.load_from_csv(node_path, edge_path)
            else:
                st.error(f"Dosyalar Bulunamadı! Lütfen kontrol edin: {data_dir}")
        
        # Kontrol: Ag olustu mu?
        if gen.G is not None and gen.G.number_of_nodes() > 0:
            st.session_state.G = gen.G
            # Coğrafi Koordinatlara Eşle
            st.session_state.pos_geo = generate_geo_positions(gen.G, seed=int(seed_val))
            st.session_state.last_path = None
            st.session_state.last_metrics = None
            
            st.toast(f"Takımyıldız Çevrimiçi: {len(st.session_state.G.nodes())} Uydu", icon="🚀")
        else:
             st.error("Ağ Oluşturulamadı! (CSV dosyaları hatalı veya boş olabilir)")

st.sidebar.markdown("---")
st.sidebar.info("Sistem Hazır. Rota Komutları Bekleniyor.")

# ------------------------------------------------------------------------------
# ANA PANEL (MAIN DASHBOARD)
# ------------------------------------------------------------------------------
st.title("Küresel Ağ topolojisi (3D)")

# TAB YAPISI: TEKLI ANALIZ vs KARSILASTIRMA
tab_single, tab_compare = st.tabs(["🧬 Tekli Analiz & Simülasyon", "⚔️ Algoritma Karşılaştırma"])

# ==============================================================================
# TAB 1: MEVCUT ANALIZ (TEKLI)
# ==============================================================================
# ==============================================================================
# TAB 1: MEVCUT ANALIZ (TEKLI)
# ==============================================================================
with tab_single:
    if st.session_state.G is None:
        st.warning("⚠️ TAKIMYILDIZ TESPİT EDİLEMEDİ. YAN MENÜDEN KURULUMU BAŞLATIN.")
        st.image("https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop", caption="Sistem Çevrimdışı", use_column_width=True)
    else:
        # 1. TOP SECTION: CONTROLS & INPUTS
        with st.container():
            st.markdown("### 🎯 HEDEFLEME SİSTEMİ")
            all_nodes = list(st.session_state.G.nodes())
            if all_nodes:
                min_n, max_n = min(all_nodes), max(all_nodes)
                
                # Input Row: Source, Dest, BW, Algo, Action
                c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1.5, 1])
                
                with c1:
                    source_node = st.number_input("KAYNAK ID", min_value=min_n, max_value=max_n, value=min_n, key="s_sing")
                with c2:
                    dest_node = st.number_input("HEDEF ID", min_value=min_n, max_value=max_n, value=max_n, key="d_sing")
                with c3:
                    demand_req = st.slider("Bant Gen. (Mbps)", 10, 1000, 100, key="bw_sing")
                with c4:
                    algo_choice = st.selectbox("ROTA PROTOKOLÜ", ["ACO (Karınca Kolonisi)", "GA (Genetik Algoritma)", "RL (Pekiştirmeli Öğrenme)"], key="algo_select_sing")
                with c5:
                    st.write("") # Spacer for alignment
                    st.write("") 
                    run_btn = st.button("EN İYİ ROTAYI HESAPLA", type="primary", key="btn_sing", use_container_width=True)

                # Weights Row (Collapsible or just small)
                with st.expander("⚖️ Öncelik Ağırlıkları (Gelişmiş Ayarlar)", expanded=False):
                    wc1, wc2, wc3 = st.columns(3)
                    with wc1: w_delay = st.slider("Gecikme Önceliği", 0.0, 1.0, 0.33, key="wd_sing")
                    with wc2: w_rel = st.slider("Güvenilirlik Önceliği", 0.0, 1.0, 0.33, key="wr_sing")
                    with wc3: w_res = st.slider("Maliyet Önceliği", 0.0, 1.0, 0.33, key="wc_sing")

        # 2. MIDDLE SECTION: CALCULATION LOGIC & RESULTS
        if run_btn:
            with st.spinner("Yörünge Hesaplamaları Yapılıyor..."):
                start_t = time.time()
                best_path = None
                best_val = 0
                algo_name = "Bilinmiyor"
                
                # Algoritma Seçimi
                if "ACO" in algo_choice:
                    aco = AntColonyOptimization(
                        st.session_state.G, num_ants=25, generations=25,
                        w_delay=w_delay, w_reliability=w_rel, w_resource=w_res
                    )
                    best_path, best_val = aco.run(source_node, dest_node, demand_mbps=demand_req)
                    algo_name = "ACO Sürü Zekası"
                    
                elif "GA" in algo_choice: # GA
                    ga = GeneticAlgorithm(
                        st.session_state.G, population_size=30, generations=30,
                        w_delay=w_delay, w_reliability=w_rel, w_resource=w_res
                    )
                    best_path, best_val = ga.run(source_node, dest_node, demand=demand_req)
                    algo_name = "Genetik"

                elif "RL" in algo_choice: # RL
                        rl = QLearningAlgorithm(
                        st.session_state.G, episodes=500, # Web hızı için düşük episod
                        w_delay=w_delay, w_reliability=w_rel, w_resource=w_res
                        )
                        best_path, best_val = rl.run(source_node, dest_node, demand_bw=demand_req)
                        algo_name = "Q-Learning YZ"
                
                dur = time.time() - start_t
                
                if best_path:
                    st.session_state.last_path = best_path
                    
                    # Detaylı Metrik Hesaplama
                    calc = CostCalculator(st.session_state.G, w_delay, w_rel, w_res)
                    d_total, r_cost, res_cost = calc.calculate_path_metrics(best_path)
                    
                    # Toplam Maliyet (Weight * Value)
                    total_fitness = calc.calculate_total_fitness(best_path)
                    
                    # Minimum Bant Genişliği
                    min_bw = float('inf')
                    path_len = len(best_path)
                    if path_len > 1:
                        for i in range(path_len - 1):
                            u, v = best_path[i], best_path[i+1]
                            edge_bw = st.session_state.G.edges[u, v].get('capacity_mbps', 0)
                            if edge_bw < min_bw:
                                min_bw = edge_bw
                    else:
                        min_bw = 0

                    st.session_state.last_metrics = {
                        "algo": algo_name,
                        "time": dur,
                        "path": best_path,
                        # Values
                        "min_bw": min_bw,
                        "total_delay": d_total,
                        "total_reliability": math.exp(-r_cost) if r_cost != float('inf') else 0,
                        # Costs
                        "reliability_cost": r_cost,
                        "resource_cost": res_cost,
                        "total_cost": total_fitness
                    }
                    st.toast("Rota Hesaplandı!", icon="✅")
                else:
                    st.session_state.last_path = None
                    st.session_state.last_metrics = None

        # 3. RESULTS PANEL (TOP SECTION)
        if st.session_state.last_metrics:
            m = st.session_state.last_metrics
            st.markdown("""
            <div style="background: rgba(0, 50, 0, 0.4); padding: 15px; border-radius: 8px; border: 1px solid #00ff00; margin-top: 20px; margin-bottom: 20px;">
                <h4 style="margin-top:0; border-bottom:1px solid rgba(255,255,255,0.2); padding-bottom:10px; color:#00ff00;">🚀 Sonuç Raporu</h4>
            """, unsafe_allow_html=True)
            
            c_res1, c_res2 = st.columns([1, 2])
            
            with c_res1:
                st.write(f"**Algoritma:** {m['algo']}")
                st.write(f"**Hesap Süresi:** {m['time']:.4f} sn")
                st.write("---")
                st.write(f"⭐ **TOPLAM MALİYET:** {m['total_cost']:.4f}")

            with c_res2:
                # Metrics Row
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Min. Bant Gen.", f"{m['min_bw']} Mbps")
                m2.metric("Toplam Gecikme", f"{m['total_delay']:.2f} ms")
                m3.metric("Güvenilirlik", f"{m['total_reliability']:.6f}")
                m4.metric("Kaynak Mal.", f"{m['resource_cost']:.2f}")

                with st.expander("En İyi Rota (Tam Liste)", expanded=False):
                    st.write(f"{m['path']}")
            
            st.markdown("</div>", unsafe_allow_html=True)
        elif st.session_state.last_path is None and run_btn:
             pass # Error already shown above
        else:
             # Initial state or cleared
             # st.info("Sonuçları görmek için rota hesaplayın.") # Optional: Don't show anything to keep top clean
             pass

        # 3. VISUALIZATION & RESULTS SPLIT
        # Globe on Left (2/3), Results Panel on Right (1/3)
        col_vis, col_stats = st.columns([2, 1])
        
        with col_vis:
            fig = create_globe_fig(st.session_state.G, st.session_state.pos_geo, st.session_state.last_path)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
        with col_stats:
            # A. AĞ İSTATİSTİKLERİ
            display_network_metrics(st.session_state.G)



# ==============================================================================
# TAB 2: ALGORITMA KARSILASTIRMA
# ==============================================================================
with tab_compare:
    st.markdown("### ⚔️ OPTİMİZASYON PROTOKOLLERİ SAVAŞI")
    
    if st.session_state.G is None:
        st.warning("Lütfen önce Yan Menüden bir ağ oluşturun.")
    else:
        # 1. Ortak Girisler
        c_i1, c_i2, c_i3 = st.columns(3)
        all_nodes = list(st.session_state.G.nodes())
        min_n, max_n = min(all_nodes), max(all_nodes)
        
        with c_i1:
            src_cmp = st.number_input("KAYNAK (Ortak)", min_n, max_n, min_n, key="src_cmp")
        with c_i2:
            dst_cmp = st.number_input("HEDEF (Ortak)", min_n, max_n, max_n, key="dst_cmp")
        with c_i3:
            bw_cmp = st.number_input("Bant Genişliği (Mbps)", 10, 1000, 100, key="bw_cmp")
            
        st.markdown("---")
        
        # 2. Algoritma Secimi
        col_algo_a, col_algo_b = st.columns(2)
        with col_algo_a:
            st.info("Algoritma A")
            algo_a = st.selectbox("Seçiniz", ["ACO", "GA", "RL"], index=0, key="algo_a")
        with col_algo_b:
            st.info("Algoritma B")
            algo_b = st.selectbox("Seçiniz", ["ACO", "GA", "RL"], index=1, key="algo_b")
            
        # 3. Karsilastir Butonu
        if st.button("🚀 KARŞILAŞTIRMAYI BAŞLAT", use_container_width=True):
            with st.spinner("Algoritmalar Yarışıyor..."):
                
                # Kosu Fonksiyonu (Local Helper)
                def run_algorithm_instance(algo_type, G, s, d, bw):
                    start = time.time()
                    path = None
                    cost = 0
                    
                    if algo_type == "ACO":
                        solver = AntColonyOptimization(G, num_ants=20, generations=20)
                        path, cost = solver.run(s, d, bw)
                    elif algo_type == "GA":
                        solver = GeneticAlgorithm(G, population_size=20, generations=20)
                        path, cost = solver.run(s, d, bw)
                    elif algo_type == "RL":
                        solver = QLearningAlgorithm(G, episodes=200) # Hiz icin dusuk
                        path, cost = solver.run(s, d, bw)
                        
                    dur = time.time() - start
                    
                    metrics = {}
                    if path:
                        calc = CostCalculator(G)
                        delay, _, _ = calc.calculate_path_metrics(path)
                        metrics = {
                            "time": dur,
                            "delay": delay,
                            "cost": cost,
                            "hops": len(path)
                        }
                    return path, metrics

                # Paralel (ardisik) Kosu
                path_a, metrics_a = run_algorithm_instance(algo_a, st.session_state.G, src_cmp, dst_cmp, bw_cmp)
                path_b, metrics_b = run_algorithm_instance(algo_b, st.session_state.G, src_cmp, dst_cmp, bw_cmp)
                
                # 4. Sonuclari Goster
                res_a, res_b = st.columns(2)
                
                # SONUC A
                with res_a:
                    st.markdown(f"#### {algo_a} Sonuçları")
                    if path_a:
                        fig_a = create_globe_fig(st.session_state.G, st.session_state.pos_geo, path_a)
                        st.plotly_chart(fig_a, use_container_width=True, key="chart_a")
                        
                        st.success(f"Süre: {metrics_a['time']:.4f}s")
                        st.info(f"Gecikme: {metrics_a['delay']:.1f}ms")
                        st.warning(f"Atlama (Hops): {metrics_a['hops']}")
                    else:
                        st.error("Rota Bulunamadı")
                        
                # SONUC B
                with res_b:
                    st.markdown(f"#### {algo_b} Sonuçları")
                    if path_b:
                        fig_b = create_globe_fig(st.session_state.G, st.session_state.pos_geo, path_b)
                        st.plotly_chart(fig_b, use_container_width=True, key="chart_b")
                        
                        st.success(f"Süre: {metrics_b['time']:.4f}s")
                        st.info(f"Gecikme: {metrics_b['delay']:.1f}ms")
                        st.warning(f"Atlama (Hops): {metrics_b['hops']}")
                    else:
                        st.error("Rota Bulunamadı")
