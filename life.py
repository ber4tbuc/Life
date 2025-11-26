"""
================================================================================
BİLİMSEL EVRİM SİMÜLASYONU - POPÜLASYON GENETİĞİ VE EVRİMSEL BİYOLOJİ
================================================================================

Bu simülasyon, gerçek evrimsel biyoloji ve popülasyon genetiği prensiplerine
dayalı bir dijital ekosistemdir. Ajanlar genomları, fitness skorları ve
genetik çeşitlilik metrikleri ile modellenmiştir.

BİLİMSEL ÖZELLİKLER:
--------------------
1. GENOM SİSTEMİ
   - Her ajanın DNA/gen dizisi (genom) vardır
   - Genotip-fenotip ilişkisi: genom → fenotip (görünüm, davranış)
   - Alel frekansları ve genetik çeşitlilik takibi

2. FİTNESS SKORU
   - Hayatta kalma başarısı (survival fitness)
   - Üreme başarısı (reproductive fitness)
   - Toplam fitness = survival × reproduction
   - Seleksiyon baskısına göre fitness değişimi

3. POPÜLASYON GENETİĞİ METRİKLERİ
   - Hardy-Weinberg dengesi testi
   - Heterozigotluk (He) ve homozigotluk (Ho)
   - F-statistics (Fst, Fis, Fit)
   - Genetik sürüklenme (genetic drift)

4. YAŞAM TABLOLARI (LIFE TABLES)
   - Yaşa özel ölüm oranları (age-specific mortality)
   - Yaşa özel doğum oranları (age-specific fecundity)
   - Net üreme oranı (R0)
   - Yaşam beklentisi (life expectancy)

5. EVRİMSEL MEKANİZMALAR
   - Doğal seleksiyon (natural selection)
   - Mutasyon (mutation)
   - Genetik sürüklenme (genetic drift)
   - Gen akışı (gene flow)

6. İSTATİSTİKSEL ANALİZLER
   - Allel frekans dağılımları
   - Fitness dağılımı
   - Popülasyon büyüme eğrisi
   - Seleksiyon katsayısı (s)

KULLANIM:
---------
- SPACE: Duraklat/Devam
- R: Simülasyonu sıfırla
- S: Detaylı bilimsel istatistikler
- CTRL+LMB: Ajan seç (genetik bilgileri görüntüle)
- LMB: Besin ekle
- RMB: Zehir ekle
- Sol/Sağ ok: Simülasyon hızını ayarla

Şuanki Versiyonda Algoritmik Problemler Var Düzeltilecek
================================================================================
"""

import pygame
import random
import math
import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # Pygame ile uyumlu backend
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import pandas as pd
from collections import deque, defaultdict
from typing import List, Tuple, Dict, Optional, Set
from itertools import product
import json
import csv
from datetime import datetime

# Renkler
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
YELLOW = (255, 255, 0)

class Agent:
    """
    Birey (Agent) sınıfı - Evrimsel simülasyondaki temel organizma birimi
    
    Her ajan şu bilimsel özelliklere sahiptir:
    - Genom: DNA/gen dizisi (genotip)
    - Fenotip: Görünüm ve davranış (genotip + çevre)
    - Fitness: Hayatta kalma ve üreme başarısı
    - Demografik özellikler: Yaş, nesil, üreme geçmişi
    """
    next_id = 1
    
    def __init__(self, x, y, parent_genome=None):
        """
        Yeni bir ajan oluştur
        
        Args:
            x, y: Başlangıç koordinatları
            parent_genome: Ebeveyn genomu (üreme için)
        """
        self.id = Agent.next_id
        Agent.next_id += 1
        self.x = x
        self.y = y
        self.energy = 200 + random.randint(0, 300)
        self.speed = 1 + random.randint(0, 2)
        self.vision = 2 + random.randint(0, 2)
        self.shape = []  # Fenotip: relatif piksel koordinatları [(x, y), ...]
        self.colors = []  # Fenotip: shape ile aynı uzunlukta [(r, g, b), ...]
        self.min_shape_size = 1
        self.generation = 0
        self.age = 0  # Yaş (tick sayısı)
        
        # ========== BİLİMSEL ÖZELLİKLER ==========
        
        # GENOM SİSTEMİ
        # Her ajanın genomu, fenotipini belirleyen gen dizisidir
        # Genom: [speed_gene, vision_gene, color_genes...]
        if parent_genome:
            # parent_genome zaten reproduce() içinde mutate edilmiş olarak geliyor
            # Burada sadece kopyala (çift mutasyon yapmamak için)
            self.genome = parent_genome.copy()
        else:
            self.genome = self.generate_random_genome()
        
        # FİTNESS METRİKLERİ
        self.fitness_survival = 0.0  # Hayatta kalma fitness'ı
        self.fitness_reproductive = 0.0  # Üreme fitness'ı
        self.fitness_total = 0.0  # Toplam fitness
        self.offspring_count = 0  # Üretilen yavru sayısı
        self.max_energy_reached = self.energy  # Ulaşılan maksimum enerji
        
        # DEMOGRAFİK VERİLER
        self.birth_tick = 0  # Doğum zamanı
        self.death_tick = None  # Ölüm zamanı (None = hala hayatta)
        self.reproduction_events = []  # Üreme olayları [(tick, offspring_id), ...]
        
        # ========== YENİ ÖZELLİKLER ==========
        
        # FİLOGENETİK AĞAÇ
        self.parent_id = None  # Ebeveyn ID
        self.children_ids = []  # Yavru ID'leri
        self.lineage_depth = 0
        
        # PERFORMANS OPTİMİZASYONU: Cache edilmiş değerler
        self._cached_color_scores = None  # (red, green, blue) scores
        self._cached_color_scores_valid = False  # Cache geçerliliği  # Soy derinliği
        
        # YAPAY ZEKA: Neural Network (genlerine göre ağırlıklar)
        self.neural_network = self._build_neural_network()
        
        # EŞEYLİ ÜREME
        self.sex = random.choice(['male', 'female'])  # Cinsiyet
        self.mate_id = None  # Eş ID (eşeyli üreme için)
        
        # ÇEVRESEL TOLERANS
        self.temperature_tolerance = random.uniform(0.3, 0.7)  # Sıcaklık toleransı
        self.ph_tolerance = random.uniform(0.4, 0.6)  # pH toleransı
        self.oxygen_tolerance = random.uniform(0.5, 0.8)  # Oksijen toleransı
        
        # ENERJİ BÜTÇESİ
        self.metabolic_rate = random.uniform(0.8, 1.2)  # Metabolik hız
        self.energy_storage = 0  # Enerji depolama
        self.energy_efficiency = random.uniform(0.7, 1.0)  # Enerji verimliliği
        
        # SOSYAL DAVRANIŞ
        self.group_id = None  # Grup ID
        self.social_score = random.uniform(0.0, 1.0)  # Sosyal eğilim
        self.cooperation_tendency = random.uniform(0.0, 1.0)  # İşbirliği eğilimi
        
        # ========== MODÜL 1: REINFORCEMENT LEARNING ==========
        # Q-learning tabanlı öğrenme sistemi
        self.q_table = {}  # {state: {action: Q_value}}
        self.learning_rate = 0.1  # Alpha - öğrenme hızı
        self.discount_factor = 0.9  # Gamma - gelecek ödüllerin değeri
        self.exploration_rate = 0.4  # Epsilon - keşif oranı
        self.memory = deque(maxlen=100)  # Son deneyimler [(state, action, reward, next_state), ...]
        self.last_state = None
        self.last_action = None
        self.total_reward = 0.0
        
        # COĞRAFİ İZOLASYON
        self.region_id = 0  # Bölge ID
        self.migration_rate = random.uniform(0.0, 0.1)  # Göç oranı
        
        # YAŞ GRUPLARI
        self.life_stage = 'juvenile'  # juvenile, adult, senescent
        
        # Fenotip oluştur (genomdan)
        self.generate_phenotype_from_genome()
    
    def generate_random_genome(self) -> List[float]:
        """
        Rastgele bir genom oluştur
        
        Genom yapısı:
        - [0]: speed_gene (0-1 arası, hızı belirler)
        - [1]: vision_gene (0-1 arası, görüşü belirler)
        - [2-11]: color_genes (her biri 0-1 arası, renkleri belirler)
        
        Returns:
            Genom dizisi
        """
        genome = [
            random.random(),  # speed_gene
            random.random(),  # vision_gene
        ]
        # 10 adet renk geni (RGB için)
        for _ in range(10):
            genome.append(random.random())
        return genome
    
    def mutate_genome(self, parent_genome: List[float], mutation_rate: float = 0.2) -> List[float]:
        """
        Ebeveyn genomundan mutasyonlu yavru genomu oluştur
        
        Args:
            parent_genome: Ebeveyn genomu
            mutation_rate: Mutasyon oranı (0-1 arası)
        
        Returns:
            Mutasyonlu genom
        """
        child_genome = []
        for gene in parent_genome:
            if random.random() < mutation_rate:
                # Mutasyon: gen değerini rastgele değiştir
                mutation_strength = 0.1
                new_gene = gene + random.uniform(-mutation_strength, mutation_strength)
                new_gene = max(0.0, min(1.0, new_gene))  # 0-1 arası sınırla
                child_genome.append(new_gene)
            else:
                # Mutasyon yok, ebeveyn genini kopyala
                child_genome.append(gene)
        return child_genome
    
    def generate_phenotype_from_genome(self):
        """
        Genomdan fenotip oluştur (genotip → fenotip dönüşümü)
        
        Bu fonksiyon genomdaki genleri fenotip özelliklerine çevirir:
        - speed_gene → self.speed
        - vision_gene → self.vision
        - color_genes → self.colors ve self.shape
        """
        # Genom uzunluğu kontrolü (IndexError önleme)
        if len(self.genome) < 3:
            # Minimum genom uzunluğu yoksa varsayılan değerler kullan
            self.speed = 1
            self.vision = 2
            self.shape = [(0, 0)]
            self.colors = [(128, 128, 128)]
            return
        
        # Hız ve görüş genlerini fenotipe çevir
        self.speed = max(1, int(1 + self.genome[0] * 2))
        self.vision = max(1, int(2 + self.genome[1] * 2))
        
        # Renk genlerinden fenotip oluştur
        num_pixels = 8 + int(self.genome[2] * 7) if len(self.genome) > 2 else 8  # 8-15 piksel
        
        # İlk piksel (genom uzunluğu kontrolü ile)
        if len(self.genome) >= 6:
            base_color = (
                int(self.genome[3] * 255),
                int(self.genome[4] * 255),
                int(self.genome[5] * 255)
            )
        else:
            # Yetersiz gen varsa varsayılan renk kullan
            base_color = (128, 128, 128)
        self.add_pixel_if_not_exists((0, 0), base_color)
        
        # Diğer pikseller
        for i in range(1, num_pixels):
            last = random.choice(self.shape)
            dx = last[0] + random.randint(-1, 1)
            dy = last[1] + random.randint(-1, 1)
            
            # Renk genlerinden renk oluştur
            color_idx = 6 + (i % 4) * 3  # 6, 9, 12, 15... genlerini kullan
            if color_idx + 2 < len(self.genome):
                color = (
                    int(self.genome[color_idx] * 255),
                    int(self.genome[color_idx + 1] * 255),
                    int(self.genome[color_idx + 2] * 255)
                )
            else:
                color = self.random_color()
            
            self.add_pixel_if_not_exists((dx, dy), color)
    
    def add_pixel_if_not_exists(self, p, c):
        for existing in self.shape:
            if existing[0] == p[0] and existing[1] == p[1]:
                return
        self.shape.append(p)
        self.colors.append(c)
    
    def random_color(self):
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    def _build_neural_network(self):
        """
        Genlerine göre neural network ağırlıklarını oluştur
        
        Network yapısı:
        - Input: [food_distance, poison_distance, energy_level, age_factor, nearby_agents]
        - Hidden layer: 8 nöron
        - Output: [move_x, move_y, behavior_priority] (3 çıktı)
        
        Genler ağırlıkları belirler (evrimsel öğrenme)
        """
        # Genom uzunluğuna göre ağırlıkları belirle
        # Her gen bir ağırlık değeri olarak kullanılır
        genome_len = len(self.genome)
        
        # Input layer (5 girdi) -> Hidden layer (8 nöron)
        # Ağırlık matrisi: 5x8 = 40 ağırlık
        input_to_hidden = []
        for i in range(5 * 8):
            if i < genome_len:
                # Gen değerini -1 ile 1 arasına normalize et
                weight = (self.genome[i % genome_len] - 0.5) * 2.0
            else:
                # Yetersiz gen varsa rastgele küçük değer
                weight = random.uniform(-0.1, 0.1)
            input_to_hidden.append(weight)
        input_to_hidden = np.array(input_to_hidden).reshape(5, 8)
        
        # Hidden layer -> Output layer (3 çıktı)
        # Ağırlık matrisi: 8x3 = 24 ağırlık
        hidden_to_output = []
        for i in range(8 * 3):
            idx = (i + 40) % genome_len if genome_len > 0 else 0
            if idx < genome_len:
                weight = (self.genome[idx] - 0.5) * 2.0
            else:
                weight = random.uniform(-0.1, 0.1)
            hidden_to_output.append(weight)
        hidden_to_output = np.array(hidden_to_output).reshape(8, 3)
        
        # Bias değerleri (genlerden)
        hidden_bias = []
        for i in range(8):
            idx = (i + 64) % genome_len if genome_len > 0 else 0
            if idx < genome_len:
                hidden_bias.append((self.genome[idx] - 0.5) * 2.0)
            else:
                hidden_bias.append(random.uniform(-0.5, 0.5))
        hidden_bias = np.array(hidden_bias)
        
        output_bias = []
        for i in range(3):
            idx = (i + 72) % genome_len if genome_len > 0 else 0
            if idx < genome_len:
                output_bias.append((self.genome[idx] - 0.5) * 1.0)
            else:
                output_bias.append(random.uniform(-0.2, 0.2))
        output_bias = np.array(output_bias)
        
        return {
            'input_to_hidden': input_to_hidden,
            'hidden_to_output': hidden_to_output,
            'hidden_bias': hidden_bias,
            'output_bias': output_bias
        }
    
    def _sigmoid(self, x):
        """Sigmoid aktivasyon fonksiyonu"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))  # Clip for numerical stability
    
    def _tanh(self, x):
        """Tanh aktivasyon fonksiyonu"""
        return np.tanh(np.clip(x, -500, 500))
    
    def ai_decision(self, food_list, poison_list, agents, world_width, world_height):
        """
        Yapay zeka ile karar verme
        
        Args:
            food_list: Besin listesi
            poison_list: Zehir listesi
            agents: Diğer ajanlar
            world_width, world_height: Dünya boyutları
        
        Returns:
            (dx, dy, behavior): Hareket yönü ve davranış önceliği
        """
        # En yakın besin ve zehir bul
        best_food = None
        best_food_dist = 1000.0  # Normalize edilmiş mesafe
        nearest_poison = None
        nearest_poison_dist = 1000.0
        nearby_agent_count = 0
        
        vision_limit = max(1, self.vision * 3)  # Vision 0 ise division by zero önleme
        
        for f in food_list:
            dist = abs(f[0] - self.x) + abs(f[1] - self.y)
            if dist <= vision_limit and dist < best_food_dist:
                best_food_dist = dist
                best_food = f
        
        for p in poison_list:
            dist = abs(p[0] - self.x) + abs(p[1] - self.y)
            if dist <= vision_limit and dist < nearest_poison_dist:
                nearest_poison_dist = dist
                nearest_poison = p
        
        # Yakındaki ajan sayısı
        for a in agents:
            if a == self:
                continue
            dist = abs(a.x - self.x) + abs(a.y - self.y)
            if dist <= vision_limit:
                nearby_agent_count += 1
        
        # Input vektörü (normalize edilmiş)
        # [food_distance, poison_distance, energy_level, age_factor, nearby_agents]
        food_dist_norm = min(1.0, best_food_dist / max(vision_limit, 1)) if best_food else 1.0
        poison_dist_norm = min(1.0, nearest_poison_dist / max(vision_limit, 1)) if nearest_poison else 1.0
        energy_norm = min(1.0, self.energy / 1000.0)
        age_norm = min(1.0, self.age / 5000.0)
        nearby_norm = min(1.0, nearby_agent_count / 10.0)
        
        inputs = np.array([food_dist_norm, poison_dist_norm, energy_norm, age_norm, nearby_norm])
        
        # Forward propagation
        # Hidden layer - Neural network kontrolü
        if not self.neural_network or 'input_to_hidden' not in self.neural_network:
            # Neural network yoksa varsayılan davranış
            return 0, 0, 0.5
        
        hidden_input = np.dot(inputs, self.neural_network['input_to_hidden']) + self.neural_network['hidden_bias']
        hidden_output = self._tanh(hidden_input)  # Tanh aktivasyon
        
        # Output layer
        # Neural network kontrolü
        if not self.neural_network or 'hidden_to_output' not in self.neural_network:
            # Neural network yoksa varsayılan davranış
            return 0, 0, 0.5
        
        output_input = np.dot(hidden_output, self.neural_network['hidden_to_output']) + self.neural_network['output_bias']
        outputs = self._tanh(output_input)  # Tanh aktivasyon
        
        # Çıktıları yorumla - array boyutu kontrolü
        # outputs[0]: x yönü hareket (-1 ile 1 arası)
        # outputs[1]: y yönü hareket (-1 ile 1 arası)
        # outputs[2]: Davranış önceliği (0=besin, 1=zehir kaç, 2=saldırı, 3=gezin)
        
        # Array boyutu kontrolü ve scalar değer dönüşümü
        if len(outputs) < 3:
            # Yetersiz çıktı varsa varsayılan değerler
            dx = 0
            dy = 0
            behavior_priority = 0.5
        else:
            # Numpy scalar'ı Python float'a çevir
            dx = float(outputs[0]) * self.speed
            dy = float(outputs[1]) * self.speed
            behavior_priority = float(outputs[2])
        
        # Eğer besin veya zehir varsa, AI kararını bunlara göre ayarla
        if best_food and behavior_priority < 0.3:
            # Besine git
            dx = (best_food[0] - self.x) / max(abs(best_food[0] - self.x), 1) * self.speed
            dy = (best_food[1] - self.y) / max(abs(best_food[1] - self.y), 1) * self.speed
        elif nearest_poison and behavior_priority > 0.7:
            # Zehirden kaç
                    # BUG FIX: Division by zero koruması
                    dx_diff = nearest_poison[0] - self.x
                    dy_diff = nearest_poison[1] - self.y
                    dx = -(dx_diff / max(abs(dx_diff), 1)) * self.speed if dx_diff != 0 else 0
                    dy = -(dy_diff / max(abs(dy_diff), 1)) * self.speed if dy_diff != 0 else 0
        
        return int(dx), int(dy), behavior_priority
    
    def get_state(self, food_list, poison_list, agents, world_width, world_height):
        """
        Mevcut durumu temsil eden state vektörü oluştur (RL için)
        """
        vision_limit = max(1, self.vision * 3)
        
        # En yakın besin
        nearest_food_dist = 1.0
        for f in food_list:
            dist = abs(f[0] - self.x) + abs(f[1] - self.y)
            if dist <= vision_limit:
                nearest_food_dist = min(nearest_food_dist, dist / vision_limit)
        
        # En yakın zehir
        nearest_poison_dist = 1.0
        for p in poison_list:
            dist = abs(p[0] - self.x) + abs(p[1] - self.y)
            if dist <= vision_limit:
                nearest_poison_dist = min(nearest_poison_dist, dist / vision_limit)
        
        # Yakındaki ajan sayısı
        nearby_count = 0
        for a in agents:
            if a == self:
                continue
            dist = abs(a.x - self.x) + abs(a.y - self.y)
            if dist <= vision_limit:
                nearby_count += 1
        
        # State'i discrete hale getir (Q-table için)
        # BUG FIX: Division by zero koruması ve negatif değer kontrolü
        energy_level = min(3, max(0, int(self.energy / 200)))  # 0-3 arası, negatif koruması
        food_level = min(3, max(0, int(nearest_food_dist * 3)))  # 0-3 arası
        poison_level = min(3, max(0, int(nearest_poison_dist * 3)))  # 0-3 arası
        agent_level = min(2, max(0, nearby_count))  # 0-2 arası
        
        return (energy_level, food_level, poison_level, agent_level)
    
    def choose_action(self, state):
        """Q-learning ile aksiyon seç (epsilon-greedy)"""
        if random.random() < self.exploration_rate:
            return random.choice(['move_toward_food', 'move_away_poison', 'move_random', 'stay'])
        else:
            if state not in self.q_table:
                self.q_table[state] = {}
            actions = ['move_toward_food', 'move_away_poison', 'move_random', 'stay']
            best_action = None
            best_q = float('-inf')
            for action in actions:
                q_value = self.q_table[state].get(action, 0.0)
                if q_value > best_q:
                    best_q = q_value
                    best_action = action
            return best_action if best_action else random.choice(actions)
    
    def update_q_value(self, state, action, reward, next_state):
        """Q-learning güncelleme"""
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        
        # BUG FIX: next_state None kontrolü ve boş dict kontrolü
        max_next_q = 0.0
        if next_state is not None and next_state in self.q_table:
            if self.q_table[next_state]:  # Boş dict kontrolü
                max_next_q = max(self.q_table[next_state].values())
            else:
                max_next_q = 0.0
        
        current_q = self.q_table[state][action]
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q
        self.memory.append((state, action, reward, next_state))
        self.total_reward += reward
    
    def calculate_reward(self, previous_energy, current_energy, found_food, hit_poison):
        """Reward hesapla (RL için) - Her tick için reward ver"""
        reward = 0.0
        
        # Enerji değişimi (net kazanç)
        energy_change = current_energy - previous_energy
        
        # Besin bulma (büyük ödül)
        if found_food:
            reward += 50.0  # Besin bulma ödülü
            if energy_change > 0:
                reward += min(energy_change * 0.3, 30.0)  # Enerji kazancı bonusu
        
        # Zehir teması (büyük ceza)
        if hit_poison:
            reward -= 60.0  # Zehir teması ceza
            if energy_change < -30:
                reward -= 20.0  # Ekstra enerji kaybı cezası
        
        # Enerji durumu ödülleri (her zaman)
        if current_energy > 300:
            reward += 2.0  # Çok iyi enerji seviyesi bonusu
        elif current_energy > 200:
            reward += 1.0  # İyi enerji seviyesi bonusu
        elif current_energy < 30:
            reward -= 10.0  # Çok düşük enerji ceza
        elif current_energy < 50:
            reward -= 3.0  # Düşük enerji ceza
        
        # Hayatta kalma bonusu (her tick)
        if current_energy > 100:
            reward += 0.2  # Hayatta kalma bonusu
        else:
            reward -= 0.1  # Düşük enerji küçük ceza
        
        # Enerji değişimi (her zaman)
        if energy_change > 20:  # Enerji kazancı
            reward += min(energy_change * 0.2, 20.0)  # Pozitif değişim ödülü
        elif energy_change < -20:  # Enerji kaybı
            reward += energy_change * 0.15  # Negatif değişim cezası
        elif energy_change > 0:  # Küçük kazanç
            reward += energy_change * 0.1  # Küçük pozitif ödül
        
        return reward
    
    def move(self, food_list, world_width, world_height, poison_list, agents, current_tick):
        """
        Ajan hareket ve davranış fonksiyonu
        
        Args:
            food_list: Besin listesi
            world_width, world_height: Dünya boyutları
            poison_list: Zehir listesi
            agents: Diğer ajanlar
            current_tick: Mevcut zaman (tick)
        """
        # Yaş artışı
        self.age += 1
        
        # ========== MODÜL 1: REINFORCEMENT LEARNING ==========
        previous_energy = self.energy
        previous_state = self.last_state
        current_state = self.get_state(food_list, poison_list, agents, world_width, world_height)
        
        # İlk state'i initialize et (ilk tick için)
        if previous_state is None:
            previous_state = current_state
            self.last_state = current_state
            self.last_action = 'move_random'  # İlk aksiyon
        
        # RL'i %50 şansla kullan (daha fazla öğrenme fırsatı)
        use_rl = random.random() < 0.5  # %50 şansla RL kullan
        rl_action = None
        if use_rl:
            rl_action = self.choose_action(current_state)
        
        # Fitness güncelleme (hayatta kalma başarısı) - her 5 tick'te bir yeterli
        if current_tick % 5 == 0:
            self.update_fitness()
        
        # Davranış eğilimleri: kırmızı, yeşil, mavi oranına göre (cache kullan)
        # Empty colors list kontrolü
        if len(self.colors) == 0:
            red_score = green_score = blue_score = 0.33  # Varsayılan değerler
        else:
            cached_colors_len = self._cached_color_scores[3] if self._cached_color_scores else 0
            if not self._cached_color_scores_valid or len(self.colors) != cached_colors_len:
                red_score = sum(c[0] for c in self.colors)
                green_score = sum(c[1] for c in self.colors)
                blue_score = sum(c[2] for c in self.colors)
                total = max(1, red_score + green_score + blue_score)
                red_score /= total
                green_score /= total
                blue_score /= total
                self._cached_color_scores = (red_score, green_score, blue_score, len(self.colors))
                self._cached_color_scores_valid = True
            else:
                # Cache'den güvenli şekilde al
                if self._cached_color_scores and len(self._cached_color_scores) >= 4:
                    red_score, green_score, blue_score, _ = self._cached_color_scores
                else:
                    # Cache bozuksa yeniden hesapla
                    red_score = sum(c[0] for c in self.colors)
                    green_score = sum(c[1] for c in self.colors)
                    blue_score = sum(c[2] for c in self.colors)
                    total = max(1, red_score + green_score + blue_score)
                    red_score /= total
                    green_score /= total
                    blue_score /= total
                    self._cached_color_scores = (red_score, green_score, blue_score, len(self.colors))
                    self._cached_color_scores_valid = True
        
        # Hedef belirleme
        best_food = None
        best_food_dist = float('inf')
        nearest_poison = None
        nearest_poison_dist = float('inf')
        attack_target = None
        attack_dist = float('inf')
        
        # Foods - Optimize: Early exit ve vision limit
        vision_limit = max(1, self.vision * 3)  # Vision 0 ise division by zero önleme
        vision_sq = vision_limit * vision_limit  # Squared distance for comparison
        for f in food_list:
            dx = f[0] - self.x
            dy = f[1] - self.y
            dist = abs(dx) + abs(dy)  # Manhattan distance
            if dist <= vision_limit and dist < best_food_dist:
                best_food_dist = dist
                best_food = f
                if dist == 0:  # Early exit if found exact match
                    break
        
        # Poisons - Optimize: Early exit
        for p in poison_list:
            dx = p[0] - self.x
            dy = p[1] - self.y
            dist = abs(dx) + abs(dy)
            if dist <= vision_limit and dist < nearest_poison_dist:
                nearest_poison_dist = dist
                nearest_poison = p
                if dist == 0:  # Early exit
                    break
        
        # Agents (saldırı hedefi) - Optimize: Limit search radius
        attack_radius = 1 + self.vision // 2
        for a in agents:
            if a == self:
                continue
            dx = a.x - self.x
            dy = a.y - self.y
            dist = abs(dx) + abs(dy)
            if dist <= attack_radius and dist < attack_dist:
                attack_dist = dist
                attack_target = a
                if dist == 0:  # Early exit
                    break
        
        # Hareket kararı - YAPAY ZEKA DESTEKLİ
        # AI kararı al (genlerine göre neural network)
        ai_dx, ai_dy, behavior_priority = self.ai_decision(food_list, poison_list, agents, world_width, world_height)
        
        # AI kararını kullan (genlerine göre öğrenilmiş davranış)
        # Eski renk skorlarına dayalı davranışı da koruyoruz (hibrit yaklaşım)
        use_ai = len(self.genome) >= 20  # Yeterli gen varsa AI kullan
        
        # ========== MODÜL 1: REINFORCEMENT LEARNING - Aksiyon uygula ==========
        # RL'i daha sık kullan (öğrenme için)
        rl_used = False
        if use_rl and rl_action:
            if rl_action == 'move_toward_food' and best_food:
                # Besine git
                dist_to_food = abs(best_food[0] - self.x) + abs(best_food[1] - self.y)
                if dist_to_food <= self.vision * 3:  # Vision'ın 3 katı mesafede
                    dx = (best_food[0] - self.x) / max(abs(best_food[0] - self.x), 1) * self.speed
                    dy = (best_food[1] - self.y) / max(abs(best_food[1] - self.y), 1) * self.speed
                    self.x += int(dx)
                    self.y += int(dy)
                    rl_used = True
            elif rl_action == 'move_away_poison' and nearest_poison:
                # Zehirden kaç
                dist_to_poison = abs(nearest_poison[0] - self.x) + abs(nearest_poison[1] - self.y)
                if dist_to_poison <= self.vision * 3:  # Vision'ın 3 katı mesafede
                    # BUG FIX: Division by zero koruması
                    dx_diff = nearest_poison[0] - self.x
                    dy_diff = nearest_poison[1] - self.y
                    dx = -(dx_diff / max(abs(dx_diff), 1)) * self.speed if dx_diff != 0 else 0
                    dy = -(dy_diff / max(abs(dy_diff), 1)) * self.speed if dy_diff != 0 else 0
                    self.x += int(dx)
                    self.y += int(dy)
                    rl_used = True
            elif rl_action == 'move_random':
                # Rastgele hareket (daha sık kullan)
                if not best_food or (best_food and abs(best_food[0] - self.x) + abs(best_food[1] - self.y) > self.vision * 3):
                    self.x += random.randint(-self.speed, self.speed)
                    self.y += random.randint(-self.speed, self.speed)
                    rl_used = True
            # 'stay' aksiyonu için hareket etme
        
        # RL kullanılmadıysa normal sistemleri kullan
        if not rl_used:
            use_ai = len(self.genome) >= 20  # Yeterli gen varsa AI kullan
            if use_ai:
                # AI kararına göre hareket
                self.x += ai_dx
                self.y += ai_dy
            else:
                # Eski sistem (renk skorlarına göre) - geriye dönük uyumluluk
                if nearest_poison and red_score > 0.35:
                    # Zehirden kaç
                    dx = nearest_poison[0] - self.x
                    dy = nearest_poison[1] - self.y
                    if dx > 0:
                        self.x -= min(self.speed, dx)
                    elif dx < 0:
                        self.x += min(self.speed, -dx)
                    if dy > 0:
                        self.y -= min(self.speed, dy)
                    elif dy < 0:
                        self.y += min(self.speed, -dy)
                elif best_food and green_score > 0.20:
                    # Besine git
                    dx = best_food[0] - self.x
                    dy = best_food[1] - self.y
                    if dx > 0:
                        self.x += min(self.speed, dx)
                    elif dx < 0:
                        self.x -= min(self.speed, -dx)
                    if dy > 0:
                        self.y += min(self.speed, dy)
                    elif dy < 0:
                        self.y -= min(self.speed, -dy)
                elif attack_target and blue_score > 0.25:
                    # Saldırı
                    dx = attack_target.x - self.x
                    dy = attack_target.y - self.y
                    if dx > 0:
                        self.x += min(self.speed, dx)
                    elif dx < 0:
                        self.x -= min(self.speed, -dx)
                    if dy > 0:
                        self.y += min(self.speed, dy)
                    elif dy < 0:
                        self.y -= min(self.speed, -dy)
                else:
                    # Rastgele gezinme
                    self.x += random.randint(-self.speed, self.speed)
                    self.y += random.randint(-self.speed, self.speed)
        
        # Sınırlar
        self.x = max(0, min(world_width - 1, self.x))
        self.y = max(0, min(world_height - 1, self.y))
        
        # Yaş grubunu güncelle
        self.update_life_stage()
        
        # Enerji tüketimi (yaşlılık etkisi + metabolik hız)
        age_factor = 1.0 + (self.age / 10000.0)
        energy_cost = max(1, int(self.speed * age_factor * self.metabolic_rate))
        self.energy -= energy_cost
        
        # Enerji bütçesi: fazla enerjiyi depola
        if self.energy > 800:
            excess = self.energy - 800
            self.energy_storage += int(excess * self.energy_efficiency)
            self.energy = 800
        
        # Depolanan enerjiyi kullan (enerji düşükse)
        if self.energy < 100 and self.energy_storage > 0:
            use_storage = min(200, self.energy_storage)
            self.energy += use_storage
            self.energy_storage -= use_storage
        
        # Maksimum enerji takibi (fitness için)
        if self.energy > self.max_energy_reached:
            self.max_energy_reached = self.energy
        
        # ========== MODÜL 1: REINFORCEMENT LEARNING - Reward ve Q-update ==========
        # RL kullanıldıysa veya kullanılmaya çalışıldıysa reward hesapla ve Q-table güncelle
        # BUG FIX: previous_state None kontrolü ve state initialization düzeltildi
        if use_rl:
            # Eğer previous_state None ise, current_state'i kullan (ilk tick veya state kaybı)
            if previous_state is None:
                previous_state = current_state
            
            # Daha esnek besin/zehir tespiti
            energy_consumption = max(1, int(self.speed * (1.0 + self.age / 10000.0) * self.metabolic_rate))
            found_food = self.energy > previous_energy + energy_consumption + 10  # Besin bulundu mu?
            hit_poison = self.energy < previous_energy - energy_consumption - 30  # Zehir teması oldu mu?
            reward = self.calculate_reward(previous_energy, self.energy, found_food, hit_poison)
            
            # Q-learning güncelleme (sadece action varsa)
            # BUG FIX: last_action None ise default action kullan
            action_to_use = self.last_action if self.last_action is not None else 'move_random'
            self.update_q_value(previous_state, action_to_use, reward, current_state)
            
            # Şimdiki state ve action'ı kaydet (bir sonraki tick için)
            self.last_state = current_state
            # BUG FIX: rl_action varsa kaydet, yoksa mevcut action'ı koru
            if rl_action:
                self.last_action = rl_action
            elif self.last_action is None:
                self.last_action = 'move_random'  # Default action
        else:
            # RL kullanılmadıysa sadece state'i güncelle (action'ı koru)
            self.last_state = current_state
            # BUG FIX: RL kullanılmadığında action'ı sıfırlama, koru (öğrenme için)
        
        # Exploration rate'i zamanla azalt (sadece öğrenen ajanlar için)
        # Daha yavaş azalt (daha fazla keşif)
        if len(self.q_table) > 0:
            self.exploration_rate = max(0.1, self.exploration_rate * 0.9999)  # Min 0.1'de kal
    
    def update_fitness(self):
        """
        Fitness skorunu güncelle
        
        Fitness = Hayatta kalma başarısı × Üreme başarısı
        
        Hayatta kalma fitness'ı:
        - Yaş (daha uzun yaşamak = daha yüksek fitness)
        - Enerji seviyesi (daha yüksek enerji = daha iyi)
        - Maksimum ulaşılan enerji
        
        Üreme fitness'ı:
        - Üretilen yavru sayısı
        """
        # Hayatta kalma fitness'ı (0-1 arası normalize)
        # Logaritmik ölçek daha gerçekçi (doğrusal olmayan ilişki)
        reference_age = 1000.0
        reference_energy = 500.0
        age_component = min(1.0, math.log(1 + self.age) / math.log(1 + reference_age))
        energy_component = min(1.0, self.energy / reference_energy)
        survival_score = (age_component * 0.6 + energy_component * 0.4)  # Ağırlıklı ortalama
        self.fitness_survival = survival_score
        
        # Üreme fitness'ı (yavru sayısına göre, logaritmik ölçek)
        reference_offspring = 5.0
        reproductive_score = min(1.0, math.log(1 + self.offspring_count) / math.log(1 + reference_offspring))
        self.fitness_reproductive = reproductive_score
        
        # Toplam fitness (geometrik ortalama)
        if self.fitness_survival > 0 and self.fitness_reproductive > 0:
            self.fitness_total = math.sqrt(self.fitness_survival * self.fitness_reproductive)
        else:
            self.fitness_total = (self.fitness_survival + self.fitness_reproductive) / 2.0
    
    def update_life_stage(self):
        """Yaş grubunu güncelle"""
        if self.age < 100:
            self.life_stage = 'juvenile'
        elif self.age < 1000:
            self.life_stage = 'adult'
        else:
            self.life_stage = 'senescent'
    
    def is_alive(self):
        return self.energy > 0 and len(self.shape) >= self.min_shape_size
    
    def sexual_reproduce(self, mate, mutation_rate, current_tick):
        """
        Eşeyli üreme - İki ebeveynli üreme (crossover + mutation)
        
        Args:
            mate: Eş ajan
            mutation_rate: Mutasyon oranı
            current_tick: Mevcut zaman
        
        Returns:
            Yavru Agent veya None
        """
        if self.energy > 400 and mate.energy > 400 and self.sex != mate.sex:
            # Enerji harcama
            self.energy -= 200
            mate.energy -= 200
            self.offspring_count += 1
            mate.offspring_count += 1
            
            # Crossover (recombination)
            child_genome = []
            for i in range(len(self.genome)):
                # Rastgele hangi ebeveynden gen alınacağını seç
                if random.random() < 0.5:
                    child_genome.append(self.genome[i])
                else:
                    child_genome.append(mate.genome[i])
            
            # Mutasyon
            mutated_genome = self.mutate_genome(child_genome, mutation_rate)
            
            # Yavru oluştur
            child = Agent((self.x + mate.x) // 2, (self.y + mate.y) // 2, parent_genome=mutated_genome)
            child.parent_id = self.id  # İlk ebeveyn (ana ebeveyn)
            child.lineage_depth = max(self.lineage_depth, mate.lineage_depth) + 1
            child.generation = max(self.generation, mate.generation) + 1
            child.birth_tick = current_tick
            child.min_shape_size = self.min_shape_size
            
            # Ebeveynlerin yavru listesine ekle
            self.children_ids.append(child.id)
            mate.children_ids.append(child.id)
            
            # Üreme olayını kaydet
            self.reproduction_events.append((current_tick, child.id))
            mate.reproduction_events.append((current_tick, child.id))
            
            return child
        return None
    
    def reproduce(self, mutation_rate, current_tick):
        """
        Üreme fonksiyonu - Ebeveyn genomundan yavru oluştur
        
        Args:
            mutation_rate: Mutasyon oranı
            current_tick: Mevcut zaman
        
        Returns:
            Yavru Agent veya None (enerji yetersizse)
        """
        # Üreme için minimum enerji gereksinimi
        if self.energy > 600:
            self.energy //= 2
            self.offspring_count += 1
            
            # Yavru oluştur (ebeveyn genomundan)
            # Önce gerçek mutation_rate ile mutate et, sonra Agent oluştur
            # Böylece çift mutasyon yapılmaz
            mutated_genome = self.mutate_genome(self.genome, mutation_rate)
            child = Agent(self.x, self.y, parent_genome=mutated_genome)
            # Agent.__init__ içinde parent_genome kopyalanacak ve fenotip oluşturulacak
            child.parent_id = self.id  # Filogenetik ağaç için
            child.lineage_depth = self.lineage_depth + 1
            child.generation = self.generation + 1
            child.age = 0
            child.birth_tick = current_tick
            child.min_shape_size = self.min_shape_size
            # Fenotip zaten Agent.__init__ içinde oluşturuldu, tekrar gerek yok
            
            # Ebeveynin yavru listesine ekle
            self.children_ids.append(child.id)
            
            # Üreme olayını kaydet
            self.reproduction_events.append((current_tick, child.id))
            
            return child
        return None


class World:
    """
    Dünya (World) sınıfı - Ekosistem ve popülasyon yönetimi
    
    Bu sınıf şu bilimsel işlevleri yerine getirir:
    - Popülasyon dinamikleri
    - Genetik çeşitlilik takibi
    - Hardy-Weinberg dengesi analizi
    - Yaşam tabloları
    - Fitness dağılımı
    """
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.agents = []
        self.food = []
        self.poison = []
        self.mutation_rate = 0.20
        self.tick_count = 0
        self.generation_count = 0
        
        # Temel istatistikler
        self.population_history = deque(maxlen=200)
        self.avg_speed_history = deque(maxlen=200)
        self.avg_vision_history = deque(maxlen=200)
        self.avg_age_history = deque(maxlen=200)
        self.total_born = 0
        self.total_died = 0
        
        # ========== BİLİMSEL METRİKLER ==========
        
        # FİTNESS İSTATİSTİKLERİ
        self.avg_fitness_history = deque(maxlen=200)
        self.fitness_distribution = []  # Fitness dağılımı
        
        # GENETİK ÇEŞİTLİLİK
        self.genetic_diversity_history = deque(maxlen=200)  # Heterozigotluk
        self.allele_frequencies = {}  # Alel frekansları
        
        # YAŞAM TABLOLARI
        self.age_at_death = []  # Ölüm yaşları
        self.age_at_birth = []  # Doğum yaşları (ebeveyn)
        
        # HARDY-WEINBERG METRİKLERİ
        self.hw_expected_heterozygosity = 0.0  # Beklenen heterozigotluk
        self.hw_observed_heterozygosity = 0.0  # Gözlenen heterozigotluk
        
        # ========== YENİ ÖZELLİKLER ==========
        
        # FİLOGENETİK AĞAÇ
        self.phylogenetic_tree = {}  # {agent_id: parent_id}
        self.lineage_data = defaultdict(list)  # {lineage_depth: [agent_ids]}
        
        # SELEKSİYON KATSAYISI
        self.selection_coefficients = {}  # {locus: selection_coefficient}
        
        # GENETİK MESAFE
        self.genetic_distances = {}  # Pairwise genetic distances
        self.fst_values = {}  # Fixation index values
        
        # ÇEVRESEL DEĞİŞKENLER
        self.temperature_map = np.ones((height, width)) * 0.5  # 0-1 arası
        self.ph_map = np.ones((height, width)) * 0.5  # 0-1 arası
        self.oxygen_map = np.ones((height, width)) * 0.7  # 0-1 arası
        
        # COĞRAFİ İZOLASYON
        self.regions = {}  # {region_id: [agent_ids]}
        self.region_barriers = []  # Bariyer koordinatları
        
        # SOSYAL GRUPLAR
        self.groups = defaultdict(list)  # {group_id: [agent_ids]}
        self.next_group_id = 1
        
        # ZAMAN SERİSİ VERİLERİ
        self.genetic_diversity_time_series = deque(maxlen=500)
        self.fitness_time_series = deque(maxlen=500)
        self.population_size_time_series = deque(maxlen=500)
        
        # DENEY MODU
        self.experiment_mode = False
        self.experiment_params = {}
        self.batch_results = []
        
        # VERİ EXPORT
        self.export_data = []
        
        # REPRODUCIBILITY
        self.random_seed = None
        
        self.reset_world()
    
    def save_experiment_config(self, filename):
        """Deney konfigürasyonunu kaydet (reproducibility için)"""
        config = {
            'timestamp': datetime.now().isoformat(),
            'world_size': (self.width, self.height),
            'mutation_rate': self.mutation_rate,
            'random_seed': self.random_seed,
            'initial_population': len(self.agents),
            'parameters': self.experiment_params
        }
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_experiment_config(self, filename):
        """Deney konfigürasyonunu yükle"""
        with open(filename, 'r') as f:
            config = json.load(f)
        self.experiment_params = config.get('parameters', {})
        if 'random_seed' in config:
            self.random_seed = config['random_seed']
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
        return config
    
    def reset_world(self):
        self.agents.clear()
        self.food.clear()
        self.poison.clear()
        self.tick_count = 0
        self.generation_count = 0
        self.population_history.clear()
        self.avg_speed_history.clear()
        self.avg_vision_history.clear()
        self.avg_age_history.clear()
        self.avg_fitness_history.clear()
        self.genetic_diversity_history.clear()
        self.fitness_distribution.clear()
        self.age_at_death.clear()
        self.age_at_birth.clear()
        self.total_born = 0
        self.total_died = 0
        self.hw_expected_heterozygosity = 0.0
        self.hw_observed_heterozygosity = 0.0
        
        # Başlangıç ajanları
        for i in range(40):
            self.agents.append(Agent(
                random.randint(0, self.width - 1),
                random.randint(0, self.height - 1)
            ))
        
        # Yoğun besin alanları
        for i in range(6):
            fx = random.randint(0, max(0, self.width - 3))
            fy = random.randint(0, max(0, self.height - 3))
            for dx in range(3):
                for dy in range(3):
                    self.food.append((
                        min(self.width - 1, fx + dx),
                        min(self.height - 1, fy + dy)
                    ))
        
        # Random besinler
        for i in range(60):
            self.food.append((
                random.randint(0, self.width - 1),
                random.randint(0, self.height - 1)
            ))
        
        # Başlangıç zehir
        for i in range(20):
            self.poison.append((
                random.randint(0, self.width - 1),
                random.randint(0, self.height - 1)
            ))
    
    def add_food_cluster(self, fx, fy):
        for dx in range(3):
            for dy in range(3):
                if fx + dx < self.width and fy + dy < self.height:
                    self.food.append((fx + dx, fy + dy))
    
    def add_poison_cluster(self, fx, fy):
        for dx in range(3):
            for dy in range(3):
                if fx + dx < self.width and fy + dy < self.height:
                    self.poison.append((fx + dx, fy + dy))
    
    def update_world(self):
        self.tick_count += 1
        new_agents = []
        dead_agents = []
        
        # Otomatik besin/zehir üretimi
        if self.tick_count % 40 == 0:
            for i in range(8):
                self.food.append((
                    random.randint(0, self.width - 1),
                    random.randint(0, self.height - 1)
                ))
        
        if self.tick_count % 150 == 0:
            for i in range(6):
                self.poison.append((
                    random.randint(0, self.width - 1),
                    random.randint(0, self.height - 1)
                ))
        
        # Ajan güncelle
        for a in list(self.agents):
            a.move(self.food, self.width, self.height, self.poison, self.agents, self.tick_count)
            
            # Besinleri yeme (performans iyileştirmesi: O(n²) → O(n))
            food_to_remove = []  # List kullan (tuple'lar için set yerine)
            food_remove_set = set()  # Hızlı lookup için
            # Cache green_score calculation - empty colors list kontrolü
            if len(a.colors) == 0:
                green_score = 0.33  # Varsayılan değer
            elif a._cached_color_scores_valid and a._cached_color_scores and len(a._cached_color_scores) >= 4:
                green_score = a._cached_color_scores[1]
            else:
                total = 1 + sum(c[0] + c[1] + c[2] for c in a.colors)
                green_score = sum(c[1] for c in a.colors) / total if total > 1 else 0.33
            
            found_food_flag = False  # RL için flag
            for f in self.food:
                if f in food_remove_set:  # Skip already marked
                    continue
                if abs(a.x - f[0]) <= 1 and abs(a.y - f[1]) <= 1:
                    gain = 30 + int(green_score * 50)
                    a.energy += gain
                    food_to_remove.append(f)
                    food_remove_set.add(f)
            
            # List comprehension ile kaldır (daha verimli)
            if food_to_remove:
                self.food = [f for f in self.food if f not in food_remove_set]
            
            # Zehre temas
            hit_poison_flag = False  # RL için flag
            for p in self.poison:
                if abs(a.x - p[0]) <= 1 and abs(a.y - p[1]) <= 1:
                    if random.random() < 0.45:
                        a.energy -= 150
                        hit_poison_flag = True  # RL için
                    elif len(a.shape) > a.min_shape_size and len(a.shape) == len(a.colors):
                        # shape ve colors listelerinin uzunluklarının eşit olduğundan emin ol
                        # IndexError önleme: Liste boş olabilir kontrolü
                        if len(a.shape) > 0 and len(a.colors) > 0:
                            remove_index = random.randint(0, len(a.shape) - 1)
                            a.shape.pop(remove_index)
                            # colors listesi de aynı uzunlukta olmalı (yukarıdaki kontrol)
                            if remove_index < len(a.colors):
                                a.colors.pop(remove_index)
                            a.energy -= 30
            
            # Ajanlar arasında etkileşim - Optimize: Sadece yakın ajanları kontrol et
            # Spatial optimization: Check only agents within interaction range
            for other in self.agents:
                if other == a or not other.is_alive():
                    continue
                dx = abs(a.x - other.x)
                dy = abs(a.y - other.y)
                if dx > 1 or dy > 1:  # Early exit for distant agents
                    continue
                dist = dx + dy
                if dist <= 1:
                    # Cache color scores - empty colors list kontrolü
                    if len(a.colors) == 0:
                        blue_a = 0.33  # Varsayılan değer
                    elif a._cached_color_scores_valid and a._cached_color_scores and len(a._cached_color_scores) >= 4:
                        blue_a = a._cached_color_scores[2]
                    else:
                        total_a = max(1, sum(c[0] + c[1] + c[2] for c in a.colors))
                        blue_a = sum(c[2] for c in a.colors) / total_a if total_a > 1 else 0.33
                    
                    if len(other.colors) == 0:
                        blue_b = 0.33  # Varsayılan değer
                    elif other._cached_color_scores_valid and other._cached_color_scores and len(other._cached_color_scores) >= 4:
                        blue_b = other._cached_color_scores[2]
                    else:
                        total_b = max(1, sum(c[0] + c[1] + c[2] for c in other.colors))
                        blue_b = sum(c[2] for c in other.colors) / total_b if total_b > 1 else 0.33
                    
                    if blue_a > 0.25 and a.energy > other.energy * (0.6 + random.random() * 0.6):
                        steal = min(120, other.energy // 2)
                        other.energy -= steal
                        a.energy += steal // 2
            
            # Çoğalma (eşeyli veya eşeysiz)
            # Önce eşeyli üreme dene
            child = None  # Initialize child variable
            if random.random() < 0.3:  # %30 eşeyli üreme şansı
                # Yakındaki uygun eşi bul
                for other in list(self.agents):
                    if other == a or other.sex == a.sex:
                        continue
                    dist = abs(a.x - other.x) + abs(a.y - other.y)
                    if dist <= 3 and other.energy > 400:
                        child = a.sexual_reproduce(other, self.mutation_rate, self.tick_count)
            if child:
                new_agents.append(child)
                self.generation_count = max(self.generation_count, child.generation)
                self.total_born += 1
                self.age_at_birth.append(a.age)
                # Filogenetik ağaç güncelle
                self.phylogenetic_tree[child.id] = a.id
                self.lineage_data[child.lineage_depth].append(child.id)
                break
            # Eş bulunamadı, eşeysiz üreme
            if not child:
                    child = a.reproduce(self.mutation_rate, self.tick_count)
                    if child:
                        new_agents.append(child)
                        self.generation_count = max(self.generation_count, child.generation)
                        self.total_born += 1
                        self.age_at_birth.append(a.age)
                        # Filogenetik ağaç güncelle
                        self.phylogenetic_tree[child.id] = a.id
                        self.lineage_data[child.lineage_depth].append(child.id)
            else:
                # Eşeysiz üreme
                child = a.reproduce(self.mutation_rate, self.tick_count)
                if child:
                    new_agents.append(child)
                    self.generation_count = max(self.generation_count, child.generation)
                    self.total_born += 1
                    # Yaşam tablosu verisi
                    self.age_at_birth.append(a.age)
                    # Filogenetik ağaç güncelle
                    self.phylogenetic_tree[child.id] = a.id
                    self.lineage_data[child.lineage_depth].append(child.id)
            
            if not a.is_alive():
                dead_agents.append(a)
                self.total_died += 1
                # Yaşam tablosu verisi
                a.death_tick = self.tick_count
                self.age_at_death.append(a.age)
        
        # Ölen ajanların yerlerine besin bırak
        for dead in dead_agents:
            self.food.append((dead.x, dead.y))
        
        # Ölü ajanları kaldır - Optimize: In-place removal daha verimli
        # Önce yeni ajanları ekle, sonra ölüleri kaldır (daha az liste oluşturma)
        self.agents.extend(new_agents)
        # In-place filtering (daha verimli)
        self.agents = [a for a in self.agents if a.is_alive()]
        
        # İstatistikleri güncelle
        if len(self.agents) > 0:
            avg_speed = sum(a.speed for a in self.agents) / len(self.agents)
            avg_vision = sum(a.vision for a in self.agents) / len(self.agents)
            avg_age = sum(a.age for a in self.agents) / len(self.agents)
            avg_fitness = sum(a.fitness_total for a in self.agents) / len(self.agents)
        else:
            avg_speed = 0
            avg_vision = 0
            avg_age = 0
            avg_fitness = 0
        
        self.population_history.append(len(self.agents))
        self.avg_speed_history.append(avg_speed)
        self.avg_vision_history.append(avg_vision)
        self.avg_age_history.append(avg_age)
        self.avg_fitness_history.append(avg_fitness)
        
        # Bilimsel metrikleri güncelle - Performans optimizasyonu
        if len(self.agents) > 0:
            # Pahalı işlemleri daha az sıklıkla yap
            if self.tick_count % 5 == 0:  # Her 5 tick'te bir
                self.update_genetic_metrics()
                self.update_fitness_distribution()
            if self.tick_count % 10 == 0:  # Her 10 tick'te bir
                self.update_selection_coefficients()
                self.update_genetic_distances()
            if self.tick_count % 5 == 0:  # Her 5 tick'te bir
                self.update_social_groups()
            if self.tick_count % 3 == 0:  # Her 3 tick'te bir (daha az sıklıkla)
                self.update_environmental_effects()
            self.update_regions()  # Bu hızlı, her tick'te yapılabilir
        
        # Zaman serisi verilerini güncelle
        if len(self.agents) > 0:
            self.genetic_diversity_time_series.append(self.hw_expected_heterozygosity)
            self.fitness_time_series.append(avg_fitness)
            self.population_size_time_series.append(len(self.agents))
        
        # Temizleme
        self.compact_points(self.food, 800)
        self.compact_points(self.poison, 400)
    
    def compact_points(self, points_list, max_allowed):
        """Optimize: Daha verimli liste küçültme"""
        if len(points_list) <= max_allowed:
            return
        # Random pop yerine slice kullan (daha hızlı)
        excess = len(points_list) - max_allowed
        # Rastgele excess kadar elemanı kaldır
        if excess > 0:
            indices_to_remove = set(random.sample(range(len(points_list)), excess))
            points_list[:] = [p for i, p in enumerate(points_list) if i not in indices_to_remove]
    
    def update_genetic_metrics(self):
        """
        Genetik çeşitlilik metriklerini güncelle
        
        Hesaplanan metrikler:
        - Heterozigotluk (He): Beklenen heterozigotluk
        - Alel frekansları: Her gen lokusundaki alel dağılımı
        - Hardy-Weinberg dengesi: Popülasyonun HW dengesinde olup olmadığı
        """
        if len(self.agents) == 0:
            return
        
        # Her gen lokusu için alel frekanslarını hesapla
        # Genom uzunluğu kontrolü (IndexError önleme)
        if len(self.agents[0].genome) == 0:
            return
        
        num_loci = len(self.agents[0].genome)
        allele_freqs = []
        
        for locus in range(num_loci):
            # Bu lokustaki tüm alel değerlerini topla (index kontrolü ile)
            alleles = [agent.genome[locus] for agent in self.agents if locus < len(agent.genome)]
            
            # Division by zero kontrolü
            if len(alleles) == 0:
                continue  # Bu lokus için veri yok, atla
            
            # Alel frekansı (ortalama)
            avg_allele = sum(alleles) / len(alleles)
            allele_freqs.append(avg_allele)
            
            # Varyans (genetik çeşitlilik göstergesi)
            variance = sum((a - avg_allele) ** 2 for a in alleles) / len(alleles)
            
            # Beklenen heterozigotluk (He = 1 - Σp²)
            # Basitleştirilmiş: He ≈ 2 * p * (1-p) (iki alel varsayımı)
            p = avg_allele
            expected_het = 2 * p * (1 - p) if 0 < p < 1 else 0
            allele_freqs.append(expected_het)
        
        # Ortalama heterozigotluk (division by zero kontrolü)
        if len(allele_freqs) >= 2:
            divisor = len(allele_freqs) // 2
            if divisor > 0:
                avg_heterozygosity = sum(allele_freqs[1::2]) / divisor
                self.genetic_diversity_history.append(avg_heterozygosity)
                self.hw_expected_heterozygosity = avg_heterozygosity
    
    def update_fitness_distribution(self):
        """Fitness dağılımını güncelle"""
        if len(self.agents) == 0:
            return
        
        fitnesses = [a.fitness_total for a in self.agents]
        self.fitness_distribution = fitnesses
    
    def calculate_life_table_metrics(self) -> Dict[str, float]:
        """
        Yaşam tablosu metriklerini hesapla
        
        Returns:
            Yaşam tablosu metrikleri sözlüğü
        """
        if len(self.age_at_death) == 0:
            return {
                'mean_age_at_death': 0.0,
                'life_expectancy': 0.0,
                'net_reproductive_rate': 0.0
            }
        
        # Ortalama ölüm yaşı (division by zero kontrolü)
        if len(self.age_at_death) == 0:
            mean_age_at_death = 0.0
        else:
            mean_age_at_death = sum(self.age_at_death) / len(self.age_at_death)
        
        # Yaşam beklentisi (basitleştirilmiş)
        life_expectancy = mean_age_at_death
        
        # Net üreme oranı (R0) = Ortalama yavru sayısı
        if len(self.agents) > 0:
            avg_offspring = sum(a.offspring_count for a in self.agents) / len(self.agents)
        else:
            avg_offspring = 0.0
        
        return {
            'mean_age_at_death': mean_age_at_death,
            'life_expectancy': life_expectancy,
            'net_reproductive_rate': avg_offspring
        }
    
    def draw(self, screen, pixel_size=8):
        # Arka plan
        screen.fill(BLACK)
        
        # Besin
        for f in self.food:
            green = 150 + random.randint(0, 99)
            r = max(0, min(255, 30 + random.randint(0, 39)))
            g = max(0, min(255, green))
            pygame.draw.rect(screen, (r, g, 30), 
                           (f[0] * pixel_size, f[1] * pixel_size, pixel_size, pixel_size))
        
        # Zehir
        for p in self.poison:
            pygame.draw.rect(screen, (180, 50, 50),
                           (p[0] * pixel_size, p[1] * pixel_size, pixel_size, pixel_size))
        
        # Ajanlar
        for a in self.agents:
            # shape ve colors listelerinin uzunluklarının eşit olduğundan emin ol
            for i, p in enumerate(a.shape):
                if i < len(a.colors):
                    c = a.colors[i]
                else:
                    c = (128, 128, 128)  # Varsayılan renk
                pygame.draw.rect(screen, c,
                               ((a.x + p[0]) * pixel_size,
                                (a.y + p[1]) * pixel_size,
                                pixel_size, pixel_size))
            
            # Enerji bar
            bar_w = 20
            bx = a.x * pixel_size - bar_w // 2
            by = (a.y - 2) * pixel_size
            w = max(0, min(bar_w, int((a.energy / 800.0) * bar_w)))
            pygame.draw.rect(screen, GRAY, (bx, by, bar_w, 4))
            pygame.draw.rect(screen, YELLOW, (bx, by, w, 4))
    
    def draw_agent_highlight(self, screen, agent, pixel_size=8):
        """Seçili ajanı vurgula"""
        if not agent or not agent.shape:
            return  # Boş shape kontrolü eklendi
        
        # Ajanın etrafına vurgu çiz
        min_x = min(p[0] for p in agent.shape) + agent.x
        max_x = max(p[0] for p in agent.shape) + agent.x
        min_y = min(p[1] for p in agent.shape) + agent.y
        max_y = max(p[1] for p in agent.shape) + agent.y
        
        # Vurgu kutusu
        highlight_x = min_x * pixel_size - 2
        highlight_y = min_y * pixel_size - 2
        highlight_w = (max_x - min_x + 1) * pixel_size + 4
        highlight_h = (max_y - min_y + 1) * pixel_size + 4
        
        pygame.draw.rect(screen, (255, 255, 0), (highlight_x, highlight_y, highlight_w, highlight_h), 2)
    
    def get_agent_at(self, x, y, pixel_size=8):
        """Belirli bir ekran koordinatındaki ajanı bul"""
        world_x = x // pixel_size
        world_y = y // pixel_size
        
        for a in self.agents:
            # Ajanın tüm piksellerini kontrol et
            for px, py in a.shape:
                if (a.x + px) == world_x and (a.y + py) == world_y:
                    return a
        return None
    
    # ========== YENİ ANALİZ METODLARI ==========
    
    def update_selection_coefficients(self):
        """Seleksiyon katsayılarını hesapla (s = fitness farkı)"""
        if len(self.agents) < 2:
            return
        
        # Genom uzunluğu kontrolü (IndexError önleme)
        if len(self.agents[0].genome) == 0:
            return
        
        num_loci = len(self.agents[0].genome)
        for locus in range(num_loci):
            # Bu lokustaki gen değerlerine göre fitness karşılaştır
            high_allele_agents = [a for a in self.agents if locus < len(a.genome) and a.genome[locus] > 0.5]
            low_allele_agents = [a for a in self.agents if locus < len(a.genome) and a.genome[locus] <= 0.5]
            
            if len(high_allele_agents) > 0 and len(low_allele_agents) > 0:
                high_fitness = sum(a.fitness_total for a in high_allele_agents) / len(high_allele_agents)
                low_fitness = sum(a.fitness_total for a in low_allele_agents) / len(low_allele_agents)
                
                # Seleksiyon katsayısı: s = (w_high - w_low) / w_high
                if high_fitness > 0:
                    s = (high_fitness - low_fitness) / high_fitness
                    self.selection_coefficients[locus] = s
    
    def update_genetic_distances(self):
        """Genetik mesafeleri hesapla (Euclidean distance)"""
        if len(self.agents) < 2:
            return
        
        # Genom uzunluğu kontrolü ve tutarlılık kontrolü
        if len(self.agents[0].genome) == 0:
            return
        
        # Tüm ajanların genomlarını numpy array'e çevir (tutarlı uzunluk kontrolü)
        try:
            genomes = np.array([a.genome for a in self.agents if len(a.genome) == len(self.agents[0].genome)])
            if len(genomes) < 2:
                return
            
            # Pairwise Euclidean distance
            distances = pdist(genomes, metric='euclidean')
            distance_matrix = squareform(distances)
            
            # Ortalama genetik mesafe
            if len(distances) > 0:
                self.genetic_distances['mean'] = np.mean(distances)
                self.genetic_distances['std'] = np.std(distances)
                self.genetic_distances['matrix'] = distance_matrix
        except (ValueError, IndexError) as e:
            # Hata durumunda sessizce devam et
            pass
    
    def calculate_fst(self, region1_agents, region2_agents):
        """
        Fst (Fixation Index) hesapla - popülasyonlar arası genetik farklılık
        
        Fst = (Ht - Hs) / Ht
        Ht: Toplam heterozigotluk (birleşik popülasyonda)
        Hs: Ortalama popülasyon heterozigotluğu (her popülasyonda ayrı ayrı)
        
        Popülasyon büyüklüklerine göre ağırlıklı ortalama kullanılır.
        """
        if len(region1_agents) == 0 or len(region2_agents) == 0:
            return 0.0
        
        # Genom uzunluğu kontrolü (IndexError önleme)
        if len(region1_agents[0].genome) == 0:
            return 0.0
        
        num_loci = len(region1_agents[0].genome)
        fst_values = []
        
        # Popülasyon büyüklükleri (ağırlıklandırma için)
        n1 = len(region1_agents)
        n2 = len(region2_agents)
        n_total = n1 + n2
        
        for locus in range(num_loci):
            # Her popülasyondaki alel frekansları (index kontrolü ile)
            alleles1 = [a.genome[locus] for a in region1_agents if locus < len(a.genome)]
            alleles2 = [a.genome[locus] for a in region2_agents if locus < len(a.genome)]
            
            if len(alleles1) == 0 or len(alleles2) == 0:
                continue
            
            p1 = np.mean(alleles1)
            p2 = np.mean(alleles2)
            
            # Ağırlıklı ortalama alel frekansı (popülasyon büyüklüklerine göre)
            p_total = (n1 * p1 + n2 * p2) / n_total
            
            # Fst = (Ht - Hs) / Ht
            # Ht = 2 * p_total * (1 - p_total)  # Toplam heterozigotluk (birleşik popülasyonda)
            # Hs = (n1 * H1 + n2 * H2) / n_total  # Ağırlıklı ortalama popülasyon heterozigotluğu
            
            Ht = 2 * p_total * (1 - p_total) if 0 < p_total < 1 else 0
            H1 = 2 * p1 * (1 - p1) if 0 < p1 < 1 else 0
            H2 = 2 * p2 * (1 - p2) if 0 < p2 < 1 else 0
            Hs = (n1 * H1 + n2 * H2) / n_total if n_total > 0 else 0
            
            if Ht > 0:
                fst = (Ht - Hs) / Ht
                fst_values.append(fst)
        
        return np.mean(fst_values) if len(fst_values) > 0 else 0.0
    
    def update_social_groups(self):
        """Sosyal grupları güncelle (yakın ajanlar gruplar oluşturur)"""
        if len(self.agents) < 2:
            return
        
        # Mevcut grupları temizle
        self.groups.clear()
        self.next_group_id = 1
        
        # Tüm ajanların group_id'sini sıfırla
        for a in self.agents:
            a.group_id = None
        
        # Her ajan için grup kontrolü (performans optimizasyonu: sadece yakın ajanları kontrol et)
        for a in self.agents:
            if a.group_id is None:
                # Yeni grup oluştur
                group_id = self.next_group_id
                self.next_group_id += 1
                a.group_id = group_id
                self.groups[group_id].append(a.id)
                
                # Yakındaki ajanları gruba ekle (sadece yakın ajanları kontrol et)
                for other in self.agents:
                    if other == a or other.group_id is not None:
                        continue
                    dist = abs(a.x - other.x) + abs(a.y - other.y)
                    if dist <= 5 and abs(a.social_score - other.social_score) < 0.3:
                        other.group_id = group_id
                        self.groups[group_id].append(other.id)
    
    def update_environmental_effects(self):
        """Çevresel faktörlerin ajanlar üzerindeki etkisini güncelle"""
        for a in self.agents:
            # Sınır kontrolü (IndexError önleme)
            y_idx = min(max(0, a.y), self.height - 1)
            x_idx = min(max(0, a.x), self.width - 1)
            
            # Sıcaklık etkisi
            temp = self.temperature_map[y_idx, x_idx]
            temp_fitness = 1.0 - abs(temp - a.temperature_tolerance)
            if temp_fitness < 0.5:
                a.energy -= 10
            
            # pH etkisi
            ph = self.ph_map[y_idx, x_idx]
            ph_fitness = 1.0 - abs(ph - a.ph_tolerance)
            if ph_fitness < 0.5:
                a.energy -= 5
            
            # Oksijen etkisi
            oxygen = self.oxygen_map[y_idx, x_idx]
            if oxygen < a.oxygen_tolerance * 0.7:
                a.energy -= 15
    
    def update_regions(self):
        """Coğrafi bölgeleri güncelle"""
        # Basit bölgeleme: dünyayı 4 bölgeye ayır
        mid_x = self.width // 2
        mid_y = self.height // 2
        
        # Önce bölgeleri temizle (bellek sızıntısını önle)
        self.regions.clear()
        
        for a in self.agents:
            if a.x < mid_x and a.y < mid_y:
                a.region_id = 0
            elif a.x >= mid_x and a.y < mid_y:
                a.region_id = 1
            elif a.x < mid_x and a.y >= mid_y:
                a.region_id = 2
            else:
                a.region_id = 3
            
            if a.region_id not in self.regions:
                self.regions[a.region_id] = []
            self.regions[a.region_id].append(a.id)
    
    def get_phylogenetic_lineage(self, agent_id, max_depth=10):
        """Bir ajanın soy ağacını döndür"""
        lineage = []
        current_id = agent_id
        depth = 0
        
        while current_id in self.phylogenetic_tree and depth < max_depth:
            parent_id = self.phylogenetic_tree[current_id]
            lineage.append(parent_id)
            current_id = parent_id
            depth += 1
            # Sonsuz döngüyü önle (döngüsel referans kontrolü)
            if parent_id == agent_id:
                break
        
        return lineage
    
    def build_full_phylogenetic_tree(self, max_generations=10):
        """
        Tüm popülasyonun filogenetik ağacını oluştur
        
        Returns:
            dict: {generation: [agent_ids], ...} ve {agent_id: {'parent': id, 'children': [ids], 'generation': gen}}
        """
        tree_structure = {}
        agent_info = {}
        
        # Tüm ajanları generation'a göre grupla
        for agent in self.agents:
            gen = agent.generation
            if gen not in tree_structure:
                tree_structure[gen] = []
            tree_structure[gen].append(agent.id)
            
            # Ajan bilgilerini kaydet
            agent_info[agent.id] = {
                'parent': agent.parent_id,
                'children': agent.children_ids.copy() if hasattr(agent, 'children_ids') else [],
                'generation': gen,
                'alive': True
            }
        
        # Ölü ajanları da filogenetik ağaçtan ekle
        for child_id, parent_id in self.phylogenetic_tree.items():
            if child_id not in agent_info:
                # Ölü ajan - generation'ı parent'tan tahmin et
                parent_gen = None
                if parent_id in agent_info:
                    parent_gen = agent_info[parent_id].get('generation', 0)
                    if parent_gen is not None:
                        parent_gen += 1
                
                agent_info[child_id] = {
                    'parent': parent_id,
                    'children': [],
                    'generation': parent_gen if parent_gen is not None else 0,
                    'alive': False
                }
                
                # Generation'a ekle
                if parent_gen is not None:
                    if parent_gen not in tree_structure:
                        tree_structure[parent_gen] = []
                    tree_structure[parent_gen].append(child_id)
        
        # Children listelerini doldur
        for agent_id, info in agent_info.items():
            parent_id = info['parent']
            if parent_id and parent_id in agent_info:
                if 'children' not in agent_info[parent_id]:
                    agent_info[parent_id]['children'] = []
                agent_info[parent_id]['children'].append(agent_id)
        
        return {
            'by_generation': tree_structure,
            'agent_info': agent_info,
            'max_generation': max(tree_structure.keys()) if tree_structure else 0
        }
    
    def calculate_tree_layout(self, tree_data, panel_width, node_spacing_x=120, node_spacing_y=80):
        """
        Hiyerarşik ağaç düzeni - parent-child ilişkilerini dikkate alarak pozisyonları hesapla
        
        Returns:
            dict: {agent_id: {'x': int, 'y': int, 'generation': int, 'alive': bool, 'parent': int, 'children': [int]}}
        """
        if not tree_data or not tree_data['by_generation']:
            return {}
        
        layout = {}
        agent_info = tree_data['agent_info']
        by_gen = tree_data['by_generation']
        
        # Generation'ları sırala (en küçük = root, en büyük = leaf)
        sorted_gens = sorted(by_gen.keys())
        if not sorted_gens:
            return {}
        
        # Her generation için y pozisyonu
        y_positions = {}
        current_y = 30
        for gen in sorted_gens:
            y_positions[gen] = current_y
            current_y += node_spacing_y
        
        # Root node'ları bul (parent_id = None veya parent'ı tree'de yok)
        root_nodes = []
        for agent_id, info in agent_info.items():
            parent_id = info.get('parent')
            if not parent_id or parent_id not in agent_info:
                root_nodes.append(agent_id)
        
        # Eğer root yoksa, en eski generation'daki node'ları root yap
        if not root_nodes and sorted_gens:
            root_nodes = by_gen[sorted_gens[0]]
        
        # Her node'un subtree genişliğini hesapla (recursive)
        def calculate_subtree_width(agent_id, visited=None):
            if visited is None:
                visited = set()
            if agent_id in visited:
                return 0
            visited.add(agent_id)
            
            info = agent_info.get(agent_id, {})
            children = info.get('children', [])
            
            if not children:
                return node_spacing_x
            
            total_width = 0
            for child_id in children:
                total_width += calculate_subtree_width(child_id, visited)
            
            return max(total_width, node_spacing_x)
        
        # Her node'un pozisyonunu hesapla (recursive)
        def assign_positions(agent_id, start_x, generation, visited=None):
            if visited is None:
                visited = set()
            if agent_id in visited:
                return start_x
            visited.add(agent_id)
            
            info = agent_info.get(agent_id, {})
            children = info.get('children', [])
            gen = info.get('generation', generation)
            
            # Subtree genişliğini hesapla
            subtree_width = calculate_subtree_width(agent_id, set())
            
            # Node'u ortala
            node_x = start_x + subtree_width // 2
            
            # Layout'a ekle
            layout[agent_id] = {
                'x': node_x,
                'y': y_positions.get(gen, 0),
                'generation': gen,
                'alive': info.get('alive', False),
                'parent': info.get('parent'),
                'children': children
            }
            
            # Çocukların pozisyonlarını hesapla
            if children:
                current_x = start_x
                for child_id in children:
                    child_width = calculate_subtree_width(child_id, set())
                    assign_positions(child_id, current_x, gen + 1, visited)
                    current_x += child_width
            
            return start_x + subtree_width
        
        # Root node'lardan başla
        current_x = 20
        for root_id in root_nodes:
            subtree_width = calculate_subtree_width(root_id, set())
            assign_positions(root_id, current_x, sorted_gens[0] if sorted_gens else 0)
            current_x += subtree_width + node_spacing_x // 2
        
        # Eğer bazı node'lar layout'a eklenmediyse (orphan nodes), onları da ekle
        for gen in sorted_gens:
            for agent_id in by_gen[gen]:
                if agent_id not in layout:
                    # Orphan node - parent'ı yok veya parent layout'ta yok
                    info = agent_info.get(agent_id, {})
                    # Basit bir pozisyon ver
                    layout[agent_id] = {
                        'x': current_x,
                        'y': y_positions.get(gen, 0),
                        'generation': gen,
                        'alive': info.get('alive', False),
                        'parent': info.get('parent'),
                        'children': info.get('children', [])
                    }
                    current_x += node_spacing_x
        
        return layout
    
    def get_age_distribution(self):
        """Yaş gruplarına göre dağılım"""
        juvenile = sum(1 for a in self.agents if a.life_stage == 'juvenile')
        adult = sum(1 for a in self.agents if a.life_stage == 'adult')
        senescent = sum(1 for a in self.agents if a.life_stage == 'senescent')
        return {'juvenile': juvenile, 'adult': adult, 'senescent': senescent}
    
    def calculate_correlation_matrix(self):
        """Genler ve fenotip özellikleri arası korelasyon matrisi"""
        if len(self.agents) < 3:
            return None
        
        # Veri hazırla
        data = {
            'speed': [a.speed for a in self.agents],
            'vision': [a.vision for a in self.agents],
            'energy': [a.energy for a in self.agents],
            'age': [a.age for a in self.agents],
            'fitness': [a.fitness_total for a in self.agents],
        }
        
        # Gen lokusları için
        if len(self.agents) == 0 or len(self.agents[0].genome) == 0:
            return pd.DataFrame()
        
        num_loci = len(self.agents[0].genome)
        for i in range(min(5, num_loci)):  # İlk 5 gen
            data[f'gene_{i}'] = [a.genome[i] for a in self.agents if i < len(a.genome)]
        
        df = pd.DataFrame(data)
        return df.corr()
    
    def export_to_csv(self, filename):
        """Verileri CSV'ye export et"""
        data = []
        for a in self.agents:
            data.append({
                'id': a.id,
                'generation': a.generation,
                'age': a.age,
                'energy': a.energy,
                'speed': a.speed,
                'vision': a.vision,
                'fitness_total': a.fitness_total,
                'offspring_count': a.offspring_count,
                'x': a.x,
                'y': a.y,
                'region_id': a.region_id,
                'group_id': a.group_id,
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
    
    def export_to_json(self, filename):
        """Simülasyon durumunu JSON'a export et"""
        data = {
            'tick_count': self.tick_count,
            'generation_count': self.generation_count,
            'population_size': len(self.agents),
            'mutation_rate': self.mutation_rate,
            'genetic_diversity': self.hw_expected_heterozygosity,
            'agents': [{
                'id': a.id,
                'generation': a.generation,
                'age': a.age,
                'energy': a.energy,
                'genome': a.genome,
                'fitness': a.fitness_total,
            } for a in self.agents[:100]]  # İlk 100 ajan
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def filter_agents(self, **kwargs):
        """Ajanları filtrele"""
        filtered = list(self.agents)
        
        if 'min_fitness' in kwargs:
            filtered = [a for a in filtered if a.fitness_total >= kwargs['min_fitness']]
        if 'max_fitness' in kwargs:
            filtered = [a for a in filtered if a.fitness_total <= kwargs['max_fitness']]
        if 'min_age' in kwargs:
            filtered = [a for a in filtered if a.age >= kwargs['min_age']]
        if 'max_age' in kwargs:
            filtered = [a for a in filtered if a.age <= kwargs['max_age']]
        if 'generation' in kwargs:
            filtered = [a for a in filtered if a.generation == kwargs['generation']]
        if 'region_id' in kwargs:
            filtered = [a for a in filtered if a.region_id == kwargs['region_id']]
        if 'life_stage' in kwargs:
            filtered = [a for a in filtered if a.life_stage == kwargs['life_stage']]
        
        return filtered
    
    def find_agent_by_id(self, agent_id):
        """ID'ye göre ajan bul (sadece yaşayan ajanlar)"""
        for a in self.agents:
            if a.id == agent_id:
                return a
        return None
    
    def find_agent_by_id_including_dead(self, agent_id):
        """ID'ye göre ajan bul (ölü ajanlar için filogenetik ağaçtan)"""
        # Önce yaşayan ajanlarda ara
        agent = self.find_agent_by_id(agent_id)
        if agent:
            return agent
        # Ölü ajanlar için filogenetik ağaçtan bilgi al
        # (Not: Ölü ajanların verileri artık mevcut değil, sadece ID bilgisi var)
        return None
    
    # ========== İSTATİSTİKSEL TESTLER ==========
    
    def test_hardy_weinberg(self, locus=0, alpha=0.05):
        """
        Hardy-Weinberg dengesi için chi-square testi
        
        Not: Bu simülasyon sürekli genom değerleri (0-1) kullanır.
        HW testi normalde ayrık genotipler için tasarlanmıştır.
        Burada sürekli değerleri kategorilere ayırarak yaklaşık bir test yapıyoruz.
        
        Args:
            locus: Test edilecek gen lokusu
            alpha: Anlamlılık seviyesi (default: 0.05)
        
        Returns:
            dict: {'chi_square': float, 'p_value': float, 'df': int, 'reject_H0': bool}
        """
        if len(self.agents) < 10:
            return None
        
        # Alel frekanslarını hesapla
        alleles = [a.genome[locus] for a in self.agents if locus < len(a.genome)]
        if len(alleles) == 0:
            return None
        
        # Sürekli değerleri kategorilere ayır (0-0.33: AA, 0.33-0.67: Aa, 0.67-1: aa)
        # Bu, sürekli genom değerlerini ayrık genotiplere yaklaştırmak için bir yöntemdir
        threshold_low = 0.33
        threshold_high = 0.67
        
        # Alel frekansı: ortanca değere göre (daha istatistiksel olarak anlamlı)
        median_allele = np.median(alleles)
        p = sum(1 for a in alleles if a > median_allele) / len(alleles)  # A aleli frekansı
        q = 1 - p  # a aleli frekansı
        
        # Beklenen genotip frekansları (Hardy-Weinberg: p², 2pq, q²)
        expected_AA = p * p * len(alleles)
        expected_Aa = 2 * p * q * len(alleles)
        expected_aa = q * q * len(alleles)
        
        # Gözlenen genotip frekansları (sürekli değerleri kategorilere ayırarak)
        observed_AA = sum(1 for a in alleles if a < threshold_low)
        observed_Aa = sum(1 for a in alleles if threshold_low <= a <= threshold_high)
        observed_aa = sum(1 for a in alleles if a > threshold_high)
        
        # Chi-square testi (Yates düzeltmesi: küçük örneklemler için)
        # Minimum beklenen değer kontrolü (5'ten küçükse test geçersiz)
        if expected_AA < 5 or expected_Aa < 5 or expected_aa < 5:
            # Fisher's exact test gerekir ama basitlik için uyarı ver
            return {
                'chi_square': None,
                'p_value': None,
                'df': 1,
                'reject_H0': None,
                'allele_freq_p': p,
                'allele_freq_q': q,
                'warning': 'Expected frequencies too low for chi-square test'
            }
        
        # Chi-square testi (Yates düzeltmesi ile)
        chi_square = (abs(observed_AA - expected_AA) - 0.5) ** 2 / expected_AA
        chi_square += (abs(observed_Aa - expected_Aa) - 0.5) ** 2 / expected_Aa
        chi_square += (abs(observed_aa - expected_aa) - 0.5) ** 2 / expected_aa
        
        # p-value hesapla (1 serbestlik derecesi)
        from scipy.stats import chi2
        p_value = 1 - chi2.cdf(chi_square, df=1)
        
        return {
            'chi_square': chi_square,
            'p_value': p_value,
            'df': 1,
            'reject_H0': p_value < alpha,
            'allele_freq_p': p,
            'allele_freq_q': q,
            'note': 'Continuous values approximated as discrete genotypes'
        }
    
    def test_fitness_difference(self, group1_agents, group2_agents, test_type='t-test'):
        """
        İki grup arasında fitness farkı testi
        
        Args:
            group1_agents: Birinci grup ajanları
            group2_agents: İkinci grup ajanları
            test_type: 't-test' veya 'mann-whitney'
        
        Returns:
            dict: Test sonuçları
        """
        if len(group1_agents) < 3 or len(group2_agents) < 3:
            return None
        
        fitness1 = [a.fitness_total for a in group1_agents]
        fitness2 = [a.fitness_total for a in group2_agents]
        
        if test_type == 't-test':
            # Student's t-test
            from scipy.stats import ttest_ind
            t_stat, p_value = ttest_ind(fitness1, fitness2)
            
            return {
                'test': 't-test',
                't_statistic': t_stat,
                'p_value': p_value,
                'mean1': np.mean(fitness1),
                'mean2': np.mean(fitness2),
                'significant': p_value < 0.05
            }
        else:  # mann-whitney
            # Mann-Whitney U testi (non-parametric)
            from scipy.stats import mannwhitneyu
            u_stat, p_value = mannwhitneyu(fitness1, fitness2, alternative='two-sided')
            
            return {
                'test': 'mann-whitney',
                'u_statistic': u_stat,
                'p_value': p_value,
                'median1': np.median(fitness1),
                'median2': np.median(fitness2),
                'significant': p_value < 0.05
            }
    
    def test_trend(self, time_series_data, test_type='kendall'):
        """
        Zaman serisi trend analizi
        
        Args:
            time_series_data: Zaman serisi verisi (list veya array)
            test_type: 'kendall' (Kendall's tau) veya 'spearman'
        
        Returns:
            dict: Trend test sonuçları
        """
        if len(time_series_data) < 5:
            return None
        
        x = np.arange(len(time_series_data))
        y = np.array(time_series_data)
        
        if test_type == 'kendall':
            from scipy.stats import kendalltau
            tau, p_value = kendalltau(x, y)
            
            return {
                'test': 'kendall_tau',
                'tau': tau,
                'p_value': p_value,
                'trend': 'increasing' if tau > 0 else 'decreasing',
                'significant': p_value < 0.05
            }
        else:  # spearman
            from scipy.stats import spearmanr
            rho, p_value = spearmanr(x, y)
            
            return {
                'test': 'spearman',
                'rho': rho,
                'p_value': p_value,
                'trend': 'increasing' if rho > 0 else 'decreasing',
                'significant': p_value < 0.05
            }
    
    def calculate_effective_population_size(self):
        """
        Effective population size (Ne) hesapla
        
        Ne, genetik sürüklenme için önemli bir parametredir.
        Basitleştirilmiş hesaplama: Ne ≈ N / (1 + variance in family size)
        """
        if len(self.agents) == 0:
            return 0.0
        
        # Her ajanın yavru sayısı
        family_sizes = [a.offspring_count for a in self.agents]
        
        if len(family_sizes) == 0 or np.mean(family_sizes) == 0:
            return len(self.agents)
        
        # Variance in family size
        variance = np.var(family_sizes)
        mean_family_size = np.mean(family_sizes)
        
        # Ne = N / (1 + variance/mean^2)
        N = len(self.agents)
        if variance == 0:
            Ne = N
        else:
            Ne = N / (1 + variance / (mean_family_size ** 2))
        
        return max(1, Ne)
    
    def detect_speciation_event(self, threshold=0.3):
        """
        Türleşme olayını tespit et (genetik mesafe bazlı)
        
        Args:
            threshold: Türleşme için minimum genetik mesafe eşiği
        
        Returns:
            dict: Türleşme bilgileri
        """
        if len(self.agents) < 10:
            return None
        
        # Bölgeler arası Fst hesapla
        if len(self.regions) < 2:
            return None
        
        region_ids = list(self.regions.keys())
        max_fst = 0.0
        max_pair = None
        
        for i in range(len(region_ids)):
            for j in range(i + 1, len(region_ids)):
                region1_agents = [a for a in self.agents if a.region_id == region_ids[i]]
                region2_agents = [a for a in self.agents if a.region_id == region_ids[j]]
                
                if len(region1_agents) > 0 and len(region2_agents) > 0:
                    fst = self.calculate_fst(region1_agents, region2_agents)
                    if fst > max_fst:
                        max_fst = fst
                        max_pair = (region_ids[i], region_ids[j])
        
        if max_fst >= threshold:
            return {
                'speciation_detected': True,
                'fst': max_fst,
                'regions': max_pair,
                'threshold': threshold
            }
        else:
            return {
                'speciation_detected': False,
                'fst': max_fst,
                'threshold': threshold
            }
    
    # ========== ADAPTİF PEYZAJ (FITNESS LANDSCAPE) ==========
    
    def calculate_fitness_landscape(self, resolution=20):
        """
        Adaptif peyzaj (fitness landscape) hesapla
        
        Args:
            resolution: Grid çözünürlüğü (daha yüksek = daha detaylı)
        
        Returns:
            dict: Fitness landscape verisi
        """
        if len(self.agents) < 5:
            return None
        
        # İlk iki gen lokusunu kullan (2D peyzaj için)
        if len(self.agents[0].genome) < 2:
            return None
        
        # Grid oluştur
        x_range = np.linspace(0, 1, resolution)
        y_range = np.linspace(0, 1, resolution)
        landscape = np.zeros((resolution, resolution))
        
        # Her grid noktası için fitness hesapla
        for i, x_val in enumerate(x_range):
            for j, y_val in enumerate(y_range):
                # Bu genotip için fitness tahmin et
                # Basit model: mevcut ajanların benzer genotiplere göre ortalama fitness
                fitness_sum = 0
                count = 0
                
                for agent in self.agents:
                    if len(agent.genome) >= 2:
                        # Genotip mesafesi
                        dist = np.sqrt((agent.genome[0] - x_val)**2 + (agent.genome[1] - y_val)**2)
                        # Yakın genotiplerin fitness'ını ağırlıklandır
                        weight = np.exp(-dist * 5)  # Exponential decay
                        fitness_sum += agent.fitness_total * weight
                        count += weight
                
                if count > 0:
                    landscape[j, i] = fitness_sum / count
                else:
                    landscape[j, i] = 0
        
        return {
            'landscape': landscape,
            'x_range': x_range,
            'y_range': y_range,
            'max_fitness': np.max(landscape),
            'min_fitness': np.min(landscape)
        }
    
    def find_fitness_peaks(self, landscape_data, num_peaks=3):
        """Fitness landscape'teki peak'leri bul"""
        if landscape_data is None:
            return []
        
        landscape = landscape_data['landscape']
        peaks = []
        
        # Basit peak detection: local maxima
        for i in range(1, landscape.shape[0] - 1):
            for j in range(1, landscape.shape[1] - 1):
                val = landscape[i, j]
                # 8-komşu kontrolü
                if (val > landscape[i-1, j-1] and val > landscape[i-1, j] and
                    val > landscape[i-1, j+1] and val > landscape[i, j-1] and
                    val > landscape[i, j+1] and val > landscape[i+1, j-1] and
                    val > landscape[i+1, j] and val > landscape[i+1, j+1]):
                    peaks.append({
                        'x': landscape_data['x_range'][j],
                        'y': landscape_data['y_range'][i],
                        'fitness': val
                    })
        
        # Fitness'a göre sırala ve en yüksek N tanesini al
        peaks.sort(key=lambda p: p['fitness'], reverse=True)
        return peaks[:num_peaks]
    
    # ========== DENEY TASARIMI ==========
    
    def sensitivity_analysis(self, param_name, param_range, num_runs=5):
        """
        Parametre duyarlılık analizi
        
        Args:
            param_name: Analiz edilecek parametre ('mutation_rate', vb.)
            param_range: Parametre değer aralığı (list)
            num_runs: Her parametre değeri için çalıştırılacak simülasyon sayısı
        
        Returns:
            dict: Duyarlılık analizi sonuçları
        """
        results = []
        original_value = getattr(self, param_name, None)
        
        for param_value in param_range:
            # Parametreyi ayarla
            setattr(self, param_name, param_value)
            
            # Birden fazla run için ortalama al
            fitness_values = []
            diversity_values = []
            
            for run in range(num_runs):
                # Kısa bir simülasyon çalıştır (performans için)
                for _ in range(50):
                    self.update_world()
                
                if len(self.agents) > 0:
                    avg_fitness = np.mean([a.fitness_total for a in self.agents])
                    diversity = self.hw_expected_heterozygosity
                    fitness_values.append(avg_fitness)
                    diversity_values.append(diversity)
                
                # Reset for next run
                self.reset_world()
                # Parametreyi tekrar ayarla (reset_world mutation_rate'i sıfırlayabilir)
                setattr(self, param_name, param_value)
            
            results.append({
                'param_value': param_value,
                'avg_fitness': np.mean(fitness_values) if fitness_values else 0,
                'avg_diversity': np.mean(diversity_values) if diversity_values else 0,
                'std_fitness': np.std(fitness_values) if fitness_values else 0
            })
        
        # Orijinal değeri geri yükle
        if original_value is not None:
            setattr(self, param_name, original_value)
        
        return results
    
    def grid_search_experiment(self, param_grid, max_ticks=100):
        """
        Grid search ile parametre optimizasyonu
        
        Args:
            param_grid: {param_name: [value1, value2, ...]} formatında parametre grid'i
            max_ticks: Her kombinasyon için maksimum tick sayısı
        
        Returns:
            list: Tüm kombinasyonların sonuçları
        """
        results = []
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # Tüm kombinasyonları oluştur
        combinations = list(product(*param_values))
        
        for combo in combinations:
            # Simülasyonu çalıştır
            self.reset_world()
            
            # Parametreleri ayarla (reset_world sonrası)
            for i, param_name in enumerate(param_names):
                setattr(self, param_name, combo[i])
            
            for _ in range(max_ticks):
                self.update_world()
            
            # Sonuçları kaydet
            result = {
                'params': dict(zip(param_names, combo)),
                'final_population': len(self.agents),
                'avg_fitness': np.mean([a.fitness_total for a in self.agents]) if len(self.agents) > 0 else 0,
                'diversity': self.hw_expected_heterozygosity,
                'generation': self.generation_count
            }
            results.append(result)
        
        return results
    
    # ========== ZAMAN SERİSİ ANALİZİ ==========
    
    def detect_phase_transitions(self, time_series, window_size=20, threshold=0.1):
        """
        Faz geçişlerini (phase transitions) tespit et
        
        Args:
            time_series: Zaman serisi verisi
            window_size: Analiz penceresi boyutu
            threshold: Faz geçişi için minimum değişim eşiği
        
        Returns:
            list: Faz geçişi noktaları
        """
        if len(time_series) < window_size * 2:
            return []
        
        transitions = []
        
        for i in range(window_size, len(time_series) - window_size):
            # Önceki ve sonraki pencerelerin ortalamaları
            prev_mean = np.mean(time_series[i-window_size:i])
            next_mean = np.mean(time_series[i:i+window_size])
            
            # Değişim oranı
            change = abs(next_mean - prev_mean) / max(abs(prev_mean), 0.001)
            
            if change > threshold:
                transitions.append({
                    'index': i,
                    'time': i,
                    'change': change,
                    'prev_value': prev_mean,
                    'next_value': next_mean
                })
        
        return transitions
    
    def forecast_population(self, horizon=50, method='linear'):
        """
        Popülasyon büyümesini tahmin et
        
        Args:
            horizon: Tahmin ufku (kaç tick ileri)
            method: 'linear' veya 'exponential'
        
        Returns:
            dict: Tahmin sonuçları
        """
        if len(self.population_history) < 10:
            return None
        
        history = list(self.population_history)
        x = np.arange(len(history))
        y = np.array(history)
        
        if method == 'linear':
            # Linear regression
            coeffs = np.polyfit(x, y, 1)
            future_x = np.arange(len(history), len(history) + horizon)
            forecast = np.polyval(coeffs, future_x)
        else:  # exponential
            # Exponential fit (log space)
            # Negatif değerleri önle (kopya oluştur, orijinal array'i değiştirme)
            y_safe = np.array(y, dtype=float)
            if np.min(y_safe) <= 0:
                y_safe = y_safe + 1  # Avoid log(0)
            log_y = np.log(y_safe)
            coeffs = np.polyfit(x, log_y, 1)
            future_x = np.arange(len(history), len(history) + horizon)
            forecast = np.exp(np.polyval(coeffs, future_x))
        
        return {
            'forecast': forecast.tolist() if len(forecast) > 0 else [],
            'method': method,
            'current': history[-1] if len(history) > 0 else 0,
            'predicted_final': forecast[-1] if len(forecast) > 0 else None
        }
    
    # ========== NETWORK ANALİZİ ==========
    
    def build_migration_network(self, distance_threshold=10):
        """
        Göç ağı (migration network) oluştur
        
        Args:
            distance_threshold: İki bölge arası bağlantı için maksimum mesafe
        
        Returns:
            dict: Network verisi {nodes: [], edges: []}
        """
        if len(self.regions) < 2:
            return None
        
        nodes = []
        edges = []
        
        # Her bölge bir node
        for region_id, agent_ids in self.regions.items():
            agents_in_region = [a for a in self.agents if a.id in agent_ids]
            if len(agents_in_region) > 0:
                # Bölge merkezi
                center_x = np.mean([a.x for a in agents_in_region])
                center_y = np.mean([a.y for a in agents_in_region])
                
                nodes.append({
                    'id': region_id,
                    'x': center_x,
                    'y': center_y,
                    'size': len(agents_in_region),
                    'diversity': np.mean([a.fitness_total for a in agents_in_region]) if agents_in_region else 0
                })
        
        # Bölgeler arası bağlantılar
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes[i+1:], i+1):
                # NaN kontrolü (boş bölgeler için)
                if np.isnan(node1['x']) or np.isnan(node1['y']) or np.isnan(node2['x']) or np.isnan(node2['y']):
                    continue
                dist = np.sqrt((node1['x'] - node2['x'])**2 + (node1['y'] - node2['y'])**2)
                if dist <= distance_threshold:
                    edges.append({
                        'source': node1['id'],
                        'target': node2['id'],
                        'weight': 1.0 / (dist + 1),
                        'distance': dist
                    })
        
        return {'nodes': nodes, 'edges': edges}
    
    def build_social_network(self):
        """Sosyal ağ (social network) oluştur"""
        if len(self.agents) < 2:
            return None
        
        nodes = []
        edges = []
        
        # Her ajan bir node
        for agent in self.agents:
            nodes.append({
                'id': agent.id,
                'x': agent.x,
                'y': agent.y,
                'group_id': agent.group_id,
                'fitness': agent.fitness_total
            })
        
        # Sosyal bağlantılar (aynı gruptaki ajanlar)
        for i, agent1 in enumerate(self.agents):
            for j, agent2 in enumerate(self.agents[i+1:], i+1):
                if agent1.group_id is not None and agent1.group_id == agent2.group_id:
                    dist = abs(agent1.x - agent2.x) + abs(agent1.y - agent2.y)
                    edges.append({
                        'source': agent1.id,
                        'target': agent2.id,
                        'weight': 1.0 / (dist + 1),
                        'type': 'social'
                    })
        
        return {'nodes': nodes, 'edges': edges}
    
    def calculate_network_metrics(self, network_data):
        """Network metriklerini hesapla (degree, clustering, vb.)"""
        if network_data is None or len(network_data['edges']) == 0:
            return None
        
        nodes = network_data['nodes']
        edges = network_data['edges']
        
        # Node degree hesapla
        degrees = {}
        for node in nodes:
            degrees[node['id']] = 0
        
        for edge in edges:
            # Tüm node'lar zaten 0 ile başlatıldı, direkt artır
            if edge['source'] in degrees:
                degrees[edge['source']] += 1
            if edge['target'] in degrees:
                degrees[edge['target']] += 1
        
        # Ortalama degree
        avg_degree = np.mean(list(degrees.values())) if degrees else 0
        
        # Network density
        n = len(nodes)
        max_edges = n * (n - 1) / 2 if n > 1 else 0
        density = len(edges) / max_edges if max_edges > 0 else 0
        
        return {
            'avg_degree': avg_degree,
            'density': density,
            'num_nodes': n,
            'num_edges': len(edges),
            'degrees': degrees
        }


def draw_arrow_button(screen, x, y, size, direction, color, hover=False):
    """Ok butonu çiz (direction: 'left' veya 'right')"""
    button_rect = pygame.Rect(x, y, size, size)
    border_color = (255, 255, 255) if hover else (150, 150, 150)
    pygame.draw.rect(screen, (50, 50, 50), button_rect)
    pygame.draw.rect(screen, border_color, button_rect, 2)
    
    # Ok çiz
    center_x, center_y = x + size // 2, y + size // 2
    arrow_size = size // 3
    
    if direction == 'left':
        points = [
            (center_x + arrow_size // 2, center_y),
            (center_x - arrow_size // 2, center_y - arrow_size // 2),
            (center_x - arrow_size // 2, center_y + arrow_size // 2)
        ]
    else:  # right
        points = [
            (center_x - arrow_size // 2, center_y),
            (center_x + arrow_size // 2, center_y - arrow_size // 2),
            (center_x + arrow_size // 2, center_y + arrow_size // 2)
        ]
    
    pygame.draw.polygon(screen, color, points)


def main():
    pygame.init()
    
    WORLD_WIDTH = 190
    WORLD_HEIGHT = 80
    PIXEL_SIZE = 8
    SCREEN_WIDTH = WORLD_WIDTH * PIXEL_SIZE
    SCREEN_HEIGHT = WORLD_HEIGHT * PIXEL_SIZE + 120  # UI için ekstra alan
    
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("EvoP - Evolution Simulator")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    small_font = pygame.font.Font(None, 18)
    tiny_font = pygame.font.Font(None, 12)
    
    world = World(WORLD_WIDTH, WORLD_HEIGHT)
    
    running = True
    paused = False
    show_stats = False  # İstatistik ekranı göster/gizle
    show_phylogenetic = False  # Filogenetik ağaç göster/gizle
    phylogenetic_tab = 'selected'  # 'selected' veya 'all' - hangi sekme aktif
    phylogenetic_tree_scroll = 0  # Full tree scroll pozisyonu (dikey)
    phylogenetic_tree_scroll_x = 0  # Full tree scroll pozisyonu (yatay)
    phylogenetic_tree_zoom = 1.0  # Zoom seviyesi (1.0 = normal, <1.0 = uzaklaştır, >1.0 = yakınlaştır)
    phylogenetic_max_scroll = 0  # Max scroll değerleri (scrollbar tıklama için)
    phylogenetic_max_scroll_x = 0
    dragging_phylogenetic_scrollbar = False  # Scrollbar sürükleme durumu (dikey)
    dragging_phylogenetic_scrollbar_x = False  # Scrollbar sürükleme durumu (yatay)
    show_graphs = False  # Grafikler göster/gizle
    show_filters = False  # Filtreleme paneli göster/gizle
    show_statistical_tests = False  # İstatistiksel testler paneli
    show_fitness_landscape = False  # Adaptif peyzaj görselleştirmesi
    show_experiment_design = False  # Deney tasarımı paneli
    show_network_analysis = False  # Network analizi paneli
    selected_agent = None  # Seçili ajan
    agent_panel_scroll = 0  # Ajan bilgi paneli scroll pozisyonu
    stats_panel_scroll = 0  # İstatistik paneli scroll pozisyonu
    speed_level = 5  # 1-10 arası hız seviyesi
    filter_params = {}  # Filtreleme parametreleri
    max_delay = 1000
    min_delay = 10
    delay = int(max_delay - (speed_level - 1) * (max_delay - min_delay) / 9)  # Seviyeye göre delay hesapla
    last_update = 0
    mouse_down = False
    mouse_button = None
    
    # Hız butonları pozisyonları
    speed_control_x = 600
    speed_control_y = SCREEN_HEIGHT - 40
    button_size = 30
    left_button_rect = pygame.Rect(speed_control_x, speed_control_y, button_size, button_size)
    right_button_rect = pygame.Rect(speed_control_x + button_size + 10, speed_control_y, button_size, button_size)
    hover_left = False
    hover_right = False
    
    while running:
        current_time = pygame.time.get_ticks()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    world.reset_world()
                elif event.key == pygame.K_s:
                    show_stats = not show_stats
                    stats_panel_scroll = 0  # İstatistik ekranı açıldığında scroll'u sıfırla
                elif event.key == pygame.K_t:
                    show_phylogenetic = not show_phylogenetic  # Filogenetik ağaç
                elif event.key == pygame.K_g:
                    show_graphs = not show_graphs  # Grafikler
                elif event.key == pygame.K_f:
                    show_filters = not show_filters  # Filtreleme
                elif event.key == pygame.K_e:
                    # Export verileri
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    world.export_to_csv(f"export_{timestamp}.csv")
                    world.export_to_json(f"export_{timestamp}.json")
                    print(f"Veriler export edildi: export_{timestamp}.csv/json")
                elif event.key == pygame.K_i:
                    show_statistical_tests = not show_statistical_tests  # İstatistiksel testler
                elif event.key == pygame.K_l:
                    show_fitness_landscape = not show_fitness_landscape  # Fitness landscape
                elif event.key == pygame.K_d:
                    show_experiment_design = not show_experiment_design  # Deney tasarımı
                elif event.key == pygame.K_n:
                    show_network_analysis = not show_network_analysis  # Network analizi
            elif event.type == pygame.MOUSEWHEEL:
                # Scroll ile panelleri kaydır
                # event.y > 0 = scroll aşağı (içerik aşağı iner, scroll pozisyonu azalır)
                # event.y < 0 = scroll yukarı (içerik yukarı çıkar, scroll pozisyonu artar)
                if show_phylogenetic and phylogenetic_tab == 'all':
                    # Filogenetik ağaç scroll ve zoom
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
                        # CTRL + wheel = zoom
                        zoom_factor = 1.1 if event.y > 0 else 0.9
                        phylogenetic_tree_zoom *= zoom_factor
                        phylogenetic_tree_zoom = max(0.1, min(5.0, phylogenetic_tree_zoom))  # 0.1x - 5.0x arası
                    elif keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                        # SHIFT + wheel = yatay scroll
                        phylogenetic_tree_scroll_x -= event.y * 30
                        phylogenetic_tree_scroll_x = max(0, phylogenetic_tree_scroll_x)
                    else:
                        # Normal wheel = dikey scroll
                        phylogenetic_tree_scroll -= event.y * 30
                        phylogenetic_tree_scroll = max(0, phylogenetic_tree_scroll)
                elif show_stats:
                    # İstatistik paneli scroll
                    stats_panel_scroll -= event.y * 20  # Ters yön
                    stats_panel_scroll = max(0, stats_panel_scroll)
                elif selected_agent:
                    # Ajan bilgi paneli scroll
                    agent_panel_scroll -= event.y * 20  # Ters yön
                    agent_panel_scroll = max(0, agent_panel_scroll)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Sol tık
                    # Filogenetik ağaç sekmeleri kontrolü
                    if show_phylogenetic:
                        tab1_x = 10 + 10
                        tab2_x = tab1_x + 180 + 5
                        tab_y = 10 + 35
                        tab_w = 180
                        tab_h = 25
                        
                        if (tab1_x <= event.pos[0] <= tab1_x + tab_w and 
                            tab_y <= event.pos[1] <= tab_y + tab_h):
                            phylogenetic_tab = 'selected'
                        elif (tab2_x <= event.pos[0] <= tab2_x + tab_w and 
                              tab_y <= event.pos[1] <= tab_y + tab_h):
                            phylogenetic_tab = 'all'
                    
                    # Hız butonlarını kontrol et
                    if left_button_rect.collidepoint(event.pos):
                        speed_level = max(1, speed_level - 1)  # Sol buton azaltır
                        delay = int(max_delay - (speed_level - 1) * (max_delay - min_delay) / 9)
                    elif right_button_rect.collidepoint(event.pos):
                        speed_level = min(10, speed_level + 1)  # Sağ buton artırır
                        delay = int(max_delay - (speed_level - 1) * (max_delay - min_delay) / 9)
                    elif show_phylogenetic and phylogenetic_tab == 'all':
                        # Filogenetik ağaç scrollbar tıklama ve sürükleme kontrolü
                        panel_x_ph = 10
                        panel_y_ph = 10
                        panel_w_ph = min(800, SCREEN_WIDTH - 20)
                        panel_h_ph = min(600, SCREEN_HEIGHT - 100)
                        tab_y_ph = panel_y_ph + 35
                        tab_h_ph = 25
                        content_y_ph = tab_y_ph + tab_h_ph + 5
                        content_h_ph = panel_h_ph - (content_y_ph - panel_y_ph) - 25
                        
                        # Dikey scrollbar
                        if phylogenetic_max_scroll > 0:
                            scrollbar_w = 8
                            scrollbar_x = panel_x_ph + panel_w_ph - scrollbar_w - 5
                            scrollbar_h = content_h_ph
                            scrollbar_y = content_y_ph
                            if (scrollbar_x <= event.pos[0] <= scrollbar_x + scrollbar_w and 
                                scrollbar_y <= event.pos[1] <= scrollbar_y + scrollbar_h):
                                # Scrollbar'a tıklandı - sürükleme modunu aktif et
                                dragging_phylogenetic_scrollbar = True
                                relative_y = event.pos[1] - scrollbar_y
                                phylogenetic_tree_scroll = int((relative_y / scrollbar_h) * phylogenetic_max_scroll)
                                phylogenetic_tree_scroll = max(0, min(phylogenetic_tree_scroll, phylogenetic_max_scroll))
                        
                        # Yatay scrollbar
                        if phylogenetic_max_scroll_x > 0:
                            h_scrollbar_h = 8
                            h_scrollbar_x = panel_x_ph + 10
                            h_scrollbar_y = panel_y_ph + panel_h_ph - h_scrollbar_h - 15
                            h_scrollbar_w = panel_w_ph - 20 - (scrollbar_w + 5 if phylogenetic_max_scroll > 0 else 0)
                            if (h_scrollbar_x <= event.pos[0] <= h_scrollbar_x + h_scrollbar_w and 
                                h_scrollbar_y <= event.pos[1] <= h_scrollbar_y + h_scrollbar_h):
                                # Scrollbar'a tıklandı - sürükleme modunu aktif et
                                dragging_phylogenetic_scrollbar_x = True
                                relative_x = event.pos[0] - h_scrollbar_x
                                phylogenetic_tree_scroll_x = int((relative_x / h_scrollbar_w) * phylogenetic_max_scroll_x)
                                phylogenetic_tree_scroll_x = max(0, min(phylogenetic_tree_scroll_x, phylogenetic_max_scroll_x))
                    elif event.pos[1] < WORLD_HEIGHT * PIXEL_SIZE:  # Sadece dünya alanında
                        # Ajan seçme kontrolü (Ctrl tuşu ile)
                        keys = pygame.key.get_pressed()
                        if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
                            selected_agent = world.get_agent_at(event.pos[0], event.pos[1], PIXEL_SIZE)
                            agent_panel_scroll = 0  # Yeni ajan seçildiğinde scroll'u sıfırla
                        else:
                            mouse_down = True
                            mouse_button = 'food'
                            fx = event.pos[0] // PIXEL_SIZE
                            fy = event.pos[1] // PIXEL_SIZE
                            world.add_food_cluster(fx, fy)
                elif event.button == 3:  # Sağ tık
                    if event.pos[1] < WORLD_HEIGHT * PIXEL_SIZE:  # Sadece dünya alanında
                        mouse_down = True
                        mouse_button = 'poison'
                        fx = event.pos[0] // PIXEL_SIZE
                        fy = event.pos[1] // PIXEL_SIZE
                        world.add_poison_cluster(fx, fy)
            elif event.type == pygame.MOUSEMOTION:
                # Hover kontrolü
                hover_left = left_button_rect.collidepoint(event.pos)
                hover_right = right_button_rect.collidepoint(event.pos)
                
                # Filogenetik ağaç scrollbar sürükleme
                if dragging_phylogenetic_scrollbar and show_phylogenetic and phylogenetic_tab == 'all':
                    panel_x_ph = 10
                    panel_y_ph = 10
                    panel_w_ph = min(800, SCREEN_WIDTH - 20)
                    panel_h_ph = min(600, SCREEN_HEIGHT - 100)
                    tab_y_ph = panel_y_ph + 35
                    tab_h_ph = 25
                    content_y_ph = tab_y_ph + tab_h_ph + 5
                    content_h_ph = panel_h_ph - (content_y_ph - panel_y_ph) - 25
                    
                    scrollbar_w = 8
                    scrollbar_x = panel_x_ph + panel_w_ph - scrollbar_w - 5
                    scrollbar_h = content_h_ph
                    scrollbar_y = content_y_ph
                    
                    # Mouse pozisyonuna göre scroll pozisyonunu güncelle
                    relative_y = event.pos[1] - scrollbar_y
                    relative_y = max(0, min(relative_y, scrollbar_h))  # Scrollbar sınırları içinde
                    phylogenetic_tree_scroll = int((relative_y / scrollbar_h) * phylogenetic_max_scroll)
                    phylogenetic_tree_scroll = max(0, min(phylogenetic_tree_scroll, phylogenetic_max_scroll))
                
                if dragging_phylogenetic_scrollbar_x and show_phylogenetic and phylogenetic_tab == 'all':
                    panel_x_ph = 10
                    panel_y_ph = 10
                    panel_w_ph = min(800, SCREEN_WIDTH - 20)
                    panel_h_ph = min(600, SCREEN_HEIGHT - 100)
                    
                    h_scrollbar_h = 8
                    h_scrollbar_x = panel_x_ph + 10
                    h_scrollbar_y = panel_y_ph + panel_h_ph - h_scrollbar_h - 15
                    scrollbar_w = 8
                    h_scrollbar_w = panel_w_ph - 20 - (scrollbar_w + 5 if phylogenetic_max_scroll > 0 else 0)
                    
                    # Mouse pozisyonuna göre scroll pozisyonunu güncelle
                    relative_x = event.pos[0] - h_scrollbar_x
                    relative_x = max(0, min(relative_x, h_scrollbar_w))  # Scrollbar sınırları içinde
                    phylogenetic_tree_scroll_x = int((relative_x / h_scrollbar_w) * phylogenetic_max_scroll_x)
                    phylogenetic_tree_scroll_x = max(0, min(phylogenetic_tree_scroll_x, phylogenetic_max_scroll_x))
                
                if mouse_down:
                    fx = event.pos[0] // PIXEL_SIZE
                    fy = event.pos[1] // PIXEL_SIZE
                    if mouse_button == 'food':
                        world.add_food_cluster(fx, fy)
                    elif mouse_button == 'poison':
                        world.add_poison_cluster(fx, fy)
            elif event.type == pygame.MOUSEBUTTONUP:
                mouse_down = False
                mouse_button = None
                # Scrollbar sürükleme modunu kapat
                dragging_phylogenetic_scrollbar = False
                dragging_phylogenetic_scrollbar_x = False
        
        # Güncelleme
        if not paused and current_time - last_update >= delay:
            world.update_world()
            last_update = current_time
            
            # Seçili ajan hala hayatta mı kontrol et
            if selected_agent and (not selected_agent.is_alive() or selected_agent not in world.agents):
                selected_agent = None
        
        # Çizim
        world.draw(screen, PIXEL_SIZE)
        
        # Seçili ajanı vurgula
        if selected_agent and selected_agent.is_alive():
            world.draw_agent_highlight(screen, selected_agent, PIXEL_SIZE)
        
        # PAUSED yazısı (üstte)
        if paused:
            pause_text = font.render("PAUSED", True, (255, 0, 0))
            pause_rect = pause_text.get_rect(center=(SCREEN_WIDTH // 2, 20))
            screen.blit(pause_text, pause_rect)
        
        # UI bilgileri (aşağıda)
        info_y = SCREEN_HEIGHT - 50
        pop_text = font.render(f"Population: {len(world.agents)}", True, (255, 255, 255))
        gen_text = font.render(f"Max Generation: {world.generation_count}", True, (255, 255, 255))
        mut_text = font.render(f"Mutation: {int(world.mutation_rate * 100)}%", True, (255, 255, 255))
        
        screen.blit(pop_text, (10, info_y))
        screen.blit(gen_text, (200, info_y))
        screen.blit(mut_text, (400, info_y))
        
        # Seçili ajan bilgi paneli (sağ üst, scroll edilebilir)
        if selected_agent and selected_agent.is_alive():
            panel_x = SCREEN_WIDTH - 220
            panel_y = 5
            panel_w = 210
            panel_h = min(SCREEN_HEIGHT - 60, 400)  # Ekrana sığacak maksimum yükseklik
            
            # Transparan arka plan
            panel_surface = pygame.Surface((panel_w, panel_h))
            panel_surface.set_alpha(200)
            panel_surface.fill((30, 30, 30))
            screen.blit(panel_surface, (panel_x, panel_y))
            
            # Kenarlık
            pygame.draw.rect(screen, (150, 150, 150), (panel_x, panel_y, panel_w, panel_h), 2)
            
            # İçerik yüzeyi (scroll için)
            content_height = 0
            line_height = 18
            
            # Renk skorları hesapla
            # Empty colors list kontrolü
            if len(selected_agent.colors) == 0:
                red_score = green_score = blue_score = 0.33
                total = 1.0
            else:
                red_score = sum(c[0] for c in selected_agent.colors)
                green_score = sum(c[1] for c in selected_agent.colors)
                blue_score = sum(c[2] for c in selected_agent.colors)
                total = max(1, red_score + green_score + blue_score)
            red_pct = int((red_score / total) * 100)
            green_pct = int((green_score / total) * 100)
            blue_pct = int((blue_score / total) * 100)
            
            # ========== MODÜL 1: RL METRİKLERİ ==========
            rl_q_table_size = len(selected_agent.q_table) if hasattr(selected_agent, 'q_table') else 0
            rl_total_reward = selected_agent.total_reward if hasattr(selected_agent, 'total_reward') else 0.0
            rl_exploration = selected_agent.exploration_rate if hasattr(selected_agent, 'exploration_rate') else 0.0
            rl_memory_size = len(selected_agent.memory) if hasattr(selected_agent, 'memory') else 0
            
            info_lines = [
                f"ID: {selected_agent.id}",
                f"Gen: {selected_agent.generation}",
                f"Age: {selected_agent.age}",
                f"Energy: {int(selected_agent.energy)}",
                "",
                "=== FITNESS ===",
                f"Total: {selected_agent.fitness_total:.3f}",
                f"Survival: {selected_agent.fitness_survival:.3f}",
                f"Reproductive: {selected_agent.fitness_reproductive:.3f}",
                f"Offspring: {selected_agent.offspring_count}",
                "",
                "=== PHENOTYPE ===",
                f"Speed: {selected_agent.speed}",
                f"Vision: {selected_agent.vision}",
                f"Size: {len(selected_agent.shape)}",
                f"R:{red_pct}% G:{green_pct}% B:{blue_pct}%",
                "",
                "=== GENOME ===",
                f"Genome Length: {len(selected_agent.genome)}",
                f"Speed Gene: {selected_agent.genome[0]:.3f}" if len(selected_agent.genome) > 0 else "Speed Gene: N/A",
                f"Vision Gene: {selected_agent.genome[1]:.3f}" if len(selected_agent.genome) > 1 else "Vision Gene: N/A",
                "",
                "=== REINFORCEMENT LEARNING ===",
                f"Q-Table States: {rl_q_table_size}",
                f"Total Reward: {rl_total_reward:.1f}",
                f"Exploration Rate: {rl_exploration:.3f}",
                f"Memory Size: {rl_memory_size}",
                f"RL Active: {'Yes' if rl_q_table_size > 0 else 'No (Learning...)'}",
            ]
            
            # Toplam içerik yüksekliğini hesapla
            total_content_height = len(info_lines) * line_height + 16
            
            # Scroll limiti (içerik panelden uzunsa scroll yapılabilir)
            max_scroll = max(0, total_content_height - panel_h + 20)
            agent_panel_scroll = min(agent_panel_scroll, max_scroll)
            
            # İçeriği çiz (scroll pozisyonuna göre - S ekranındaki gibi)
            # Scroll pozisyonu = içeriğin ne kadar yukarı kaydığını gösterir
            # Scroll 0 ise: en üstteki bilgiler görünür
            # Scroll artarsa: içerik yukarı kayar (yukarıdaki bilgiler görünmez)
            y_offset = 8 - agent_panel_scroll
            for line in info_lines:
                # Sadece görünür alan içindeki satırları çiz (S ekranındaki gibi)
                if y_offset + line_height >= 0 and y_offset < panel_h - 10:
                    info_text = small_font.render(line, True, (255, 255, 255))
                    screen.blit(info_text, (panel_x + 5, panel_y + y_offset))
                y_offset += line_height
            
            # Scroll bar (eğer içerik panelden uzunsa)
            if total_content_height > panel_h:
                scrollbar_w = 4
                scrollbar_x = panel_x + panel_w - scrollbar_w - 2
                scrollbar_h = int((panel_h / total_content_height) * panel_h)
                scrollbar_y = panel_y + int((agent_panel_scroll / max_scroll) * (panel_h - scrollbar_h)) if max_scroll > 0 else panel_y
                pygame.draw.rect(screen, (100, 100, 100), (scrollbar_x, panel_y, scrollbar_w, panel_h))
                pygame.draw.rect(screen, (200, 200, 200), (scrollbar_x, scrollbar_y, scrollbar_w, scrollbar_h))
            
            # Scroll talimatı (en altta)
            if total_content_height > panel_h:
                hint_text = small_font.render("Scroll to view", True, (150, 150, 150))
                screen.blit(hint_text, (panel_x + 5, panel_y + panel_h - 15))
        
        
        # Detaylı istatistik ekranı (S tuşu ile açılır, scroll edilebilir)
        if show_stats and len(world.agents) > 0:
            # Yarı saydam arka plan
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            overlay.set_alpha(200)
            overlay.fill((0, 0, 0))
            screen.blit(overlay, (0, 0))
            
            # İstatistik paneli
            panel_x = SCREEN_WIDTH // 2 - 200
            panel_y = 50
            panel_w = 400
            panel_h = min(SCREEN_HEIGHT - 100, 500)  # Ekrana sığacak maksimum yükseklik
            
            pygame.draw.rect(screen, (40, 40, 40), (panel_x, panel_y, panel_w, panel_h))
            pygame.draw.rect(screen, (150, 150, 150), (panel_x, panel_y, panel_w, panel_h), 2)
            
            # Başlık (scroll'dan etkilenmez)
            title = font.render("STATISTICS", True, (255, 255, 255))
            screen.blit(title, (panel_x + panel_w // 2 - title.get_width() // 2, panel_y + 10))
            
            line_height = 25
            start_y = 50  # İçeriğin başlangıç y pozisyonu
            
            # Bilimsel İstatistikler
            life_table = world.calculate_life_table_metrics()
            avg_fitness = sum(a.fitness_total for a in world.agents) / len(world.agents) if len(world.agents) > 0 else 0
            genetic_div = world.genetic_diversity_history[-1] if len(world.genetic_diversity_history) > 0 else 0
            
            # ========== MODÜL 1: RL İSTATİSTİKLERİ ==========
            agents_with_rl = [a for a in world.agents if hasattr(a, 'q_table') and len(a.q_table) > 0]
            avg_rl_reward = sum(a.total_reward for a in agents_with_rl) / len(agents_with_rl) if agents_with_rl else 0.0
            avg_exploration = sum(a.exploration_rate for a in agents_with_rl) / len(agents_with_rl) if agents_with_rl else 0.0
            total_q_states = sum(len(a.q_table) for a in agents_with_rl)
            
            stats_data = [
                "=== POPULATION GENETICS ===",
                f"Population: {len(world.agents)}",
                f"Max Generation: {world.generation_count}",
                f"Genetic Diversity (He): {genetic_div:.4f}",
                f"HW Expected Het: {world.hw_expected_heterozygosity:.4f}",
                "",
                "=== FITNESS METRICS ===",
                f"Avg Fitness: {avg_fitness:.4f}",
                f"Max Fitness: {max((a.fitness_total for a in world.agents), default=0):.4f}",
                f"Min Fitness: {min((a.fitness_total for a in world.agents), default=0):.4f}",
                "",
                "=== DEMOGRAPHY ===",
                f"Total Born: {world.total_born}",
                f"Total Died: {world.total_died}",
                f"Net Reproductive Rate (R0): {life_table['net_reproductive_rate']:.2f}",
                f"Life Expectancy: {life_table['life_expectancy']:.1f} ticks",
                f"Mean Age at Death: {life_table['mean_age_at_death']:.1f}",
                "",
                "=== PHENOTYPE ===",
                f"Avg Speed: {sum(a.speed for a in world.agents) / len(world.agents):.2f}" if len(world.agents) > 0 else "Avg Speed: 0.00",
                f"Avg Vision: {sum(a.vision for a in world.agents) / len(world.agents):.2f}" if len(world.agents) > 0 else "Avg Vision: 0.00",
                f"Avg Age: {sum(a.age for a in world.agents) / len(world.agents):.0f}" if len(world.agents) > 0 else "Avg Age: 0",
                f"Avg Energy: {sum(a.energy for a in world.agents) / len(world.agents):.0f}" if len(world.agents) > 0 else "Avg Energy: 0",
                "",
                "=== REINFORCEMENT LEARNING ===",
                f"Agents with RL: {len(agents_with_rl)} / {len(world.agents)}",
                f"Avg Total Reward: {avg_rl_reward:.1f}",
                f"Avg Exploration: {avg_exploration:.3f}",
                f"Total Q-States: {total_q_states}",
                f"RL Coverage: {len(agents_with_rl) / len(world.agents) * 100:.1f}%" if len(world.agents) > 0 else "RL Coverage: 0%",
            ]
            
            # Toplam içerik yüksekliğini hesapla
            total_content_height = len(stats_data) * line_height + 20
            
            # Scroll limiti
            max_scroll = max(0, total_content_height - (panel_h - start_y) + 20)
            stats_panel_scroll = min(stats_panel_scroll, max_scroll)
            
            # İçeriği çiz (scroll pozisyonuna göre)
            y_offset = start_y - stats_panel_scroll
            for stat in stats_data:
                # Sadece görünür alan içindeki satırları çiz
                if y_offset + line_height >= start_y and y_offset < panel_h - 10:
                    if stat:
                        stat_text = small_font.render(stat, True, (255, 255, 255))
                        screen.blit(stat_text, (panel_x + 20, panel_y + y_offset))
                y_offset += line_height
            
            # Scroll bar (eğer içerik panelden uzunsa)
            if total_content_height > (panel_h - start_y):
                scrollbar_w = 4
                scrollbar_x = panel_x + panel_w - scrollbar_w - 2
                scrollbar_area_h = panel_h - start_y - 20
                scrollbar_h = int((scrollbar_area_h / total_content_height) * scrollbar_area_h)
                scrollbar_y = panel_y + start_y + int((stats_panel_scroll / max_scroll) * (scrollbar_area_h - scrollbar_h)) if max_scroll > 0 else panel_y + start_y
                pygame.draw.rect(screen, (100, 100, 100), (scrollbar_x, panel_y + start_y, scrollbar_w, scrollbar_area_h))
                pygame.draw.rect(screen, (200, 200, 200), (scrollbar_x, scrollbar_y, scrollbar_w, scrollbar_h))
            
            # Scroll talimatı (en altta)
            if total_content_height > (panel_h - start_y):
                hint_text = small_font.render("Scroll to view all", True, (150, 150, 150))
                screen.blit(hint_text, (panel_x + 20, panel_y + panel_h - 25))
            
            # Kapatma talimatı (en altta)
            close_text = small_font.render("Press S to close", True, (150, 150, 150))
            screen.blit(close_text, (panel_x + panel_w // 2 - close_text.get_width() // 2, panel_y + panel_h - 25))
            
            # Grafikler (sağ üste taşındı, scroll'dan etkilenmez)
            graph_x = SCREEN_WIDTH - 200
            graph_y = 30  # Daha aşağı indirildi
            graph_w = 180
            graph_h = 60
            
            # Speed grafiği (sağ üstte)
            if len(world.avg_speed_history) > 1:
                pygame.draw.rect(screen, (20, 20, 20), (graph_x, graph_y, graph_w, graph_h))
                pygame.draw.rect(screen, (100, 100, 100), (graph_x, graph_y, graph_w, graph_h), 1)
                max_speed = max(world.avg_speed_history) if max(world.avg_speed_history) > 0 else 1
                points = []
                for i, speed in enumerate(world.avg_speed_history):
                    x = graph_x + int((i / (len(world.avg_speed_history) - 1)) * graph_w) if len(world.avg_speed_history) > 1 else graph_x
                    y = graph_y + graph_h - int((speed / max_speed) * graph_h)
                    points.append((x, y))
                if len(points) > 1:
                    pygame.draw.lines(screen, (255, 200, 100), False, points, 2)
                speed_label = small_font.render("Speed", True, (255, 200, 100))
                screen.blit(speed_label, (graph_x + 5, graph_y - 15))
        
        # Hız kontrolleri
        speed_label = small_font.render("Speed:", True, (255, 255, 255))
        screen.blit(speed_label, (speed_control_x, speed_control_y - 20))
        
        # Ok butonları (sadece görsel ters - sol buton sağ ok, sağ buton sol ok)
        draw_arrow_button(screen, speed_control_x, speed_control_y, button_size, 'right', 
                         (255, 255, 255), hover_left)  # Sol buton sağ ok görseli (azaltır)
        draw_arrow_button(screen, speed_control_x + button_size + 10, speed_control_y, button_size, 'left', 
                         (255, 255, 255), hover_right)  # Sağ buton sol ok görseli (artırır)
        
        # Hız bar (butonların sağında)
        bar_width = 80
        bar_height = 6
        bar_x = speed_control_x + button_size * 2 + 25
        bar_y = speed_control_y + button_size // 2 - bar_height // 2
        bar_progress = (speed_level - 1) / 9.0  # 0.0 - 1.0 arası
        
        pygame.draw.rect(screen, (40, 40, 40), (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(screen, (100, 255, 100), (bar_x, bar_y, int(bar_width * bar_progress), bar_height))
        pygame.draw.rect(screen, (150, 150, 150), (bar_x, bar_y, bar_width, bar_height), 1)
        
        # Hız sayısı (barın sağında)
        speed_display = font.render(f"{speed_level}x", True, (100, 255, 100))
        speed_text_x = bar_x + bar_width + 8
        speed_text_y = speed_control_y + button_size // 2 - speed_display.get_height() // 2
        screen.blit(speed_display, (speed_text_x, speed_text_y))
        
        # Yeni özellikler UI'ı
        # Filogenetik ağaç gösterimi (sekme sistemi ile)
        if show_phylogenetic:
            # Panel arka planı (geniş panel)
            panel_x = 10
            panel_y = 10
            panel_w = min(800, SCREEN_WIDTH - 20)  # Geniş panel
            panel_h = min(600, SCREEN_HEIGHT - 100)
            
            panel_surface = pygame.Surface((panel_w, panel_h))
            panel_surface.set_alpha(220)
            panel_surface.fill((20, 20, 40))
            screen.blit(panel_surface, (panel_x, panel_y))
            pygame.draw.rect(screen, (100, 100, 150), (panel_x, panel_y, panel_w, panel_h), 2)
            
            title = small_font.render("PHYLOGENETIC TREE", True, (255, 255, 255))
            screen.blit(title, (panel_x + 10, panel_y + 10))
            
            # Sekmeler
            tab_height = 25
            tab_y = panel_y + 35
            tab_w = 180
            tab_h = tab_height
            
            # "Selected Agent" sekmesi
            tab1_x = panel_x + 10
            tab1_active = phylogenetic_tab == 'selected'
            tab1_color = (80, 80, 120) if tab1_active else (40, 40, 60)
            tab1_rect = pygame.Rect(tab1_x, tab_y, tab_w, tab_h)
            pygame.draw.rect(screen, tab1_color, tab1_rect)
            pygame.draw.rect(screen, (150, 150, 200) if tab1_active else (100, 100, 100), tab1_rect, 2)
            tab1_text = small_font.render("Selected Agent", True, (255, 255, 255) if tab1_active else (150, 150, 150))
            screen.blit(tab1_text, (tab1_x + 5, tab_y + 5))
            
            # "Full Tree" sekmesi
            tab2_x = tab1_x + tab_w + 5
            tab2_active = phylogenetic_tab == 'all'
            tab2_color = (80, 80, 120) if tab2_active else (40, 40, 60)
            tab2_rect = pygame.Rect(tab2_x, tab_y, tab_w, tab_h)
            pygame.draw.rect(screen, tab2_color, tab2_rect)
            pygame.draw.rect(screen, (150, 150, 200) if tab2_active else (100, 100, 100), tab2_rect, 2)
            tab2_text = small_font.render("Full Tree", True, (255, 255, 255) if tab2_active else (150, 150, 150))
            screen.blit(tab2_text, (tab2_x + 5, tab_y + 5))
            
            # İçerik alanı
            content_y = tab_y + tab_h + 5
            content_h = panel_h - (content_y - panel_y) - 25
            
            if phylogenetic_tab == 'selected':
                # Seçili ajan sekmesi
                y_pos = content_y
                if selected_agent:
                    lineage = world.get_phylogenetic_lineage(selected_agent.id, max_depth=10)
                    
                    # Seçili ajan
                    text = small_font.render(f"Selected: ID {selected_agent.id} (Gen {selected_agent.generation})", True, (255, 255, 100))
                    screen.blit(text, (panel_x + 10, y_pos))
                    y_pos += 25
                    
                    if lineage:
                        text = small_font.render("Ancestors:", True, (200, 200, 255))
                        screen.blit(text, (panel_x + 10, y_pos))
                        y_pos += 20
                        
                        for i, parent_id in enumerate(lineage):
                            parent = world.find_agent_by_id(parent_id)
                            if parent:
                                text = small_font.render(f"  {i+1}. ID {parent_id} (Gen {parent.generation}, Age {parent.age})", True, (180, 180, 255))
                                screen.blit(text, (panel_x + 10, y_pos))
                                y_pos += 18
                            else:
                                text = small_font.render(f"  {i+1}. ID {parent_id} (extinct)", True, (150, 150, 150))
                                screen.blit(text, (panel_x + 10, y_pos))
                                y_pos += 18
                    else:
                        text = small_font.render("No ancestors found", True, (150, 150, 150))
                        screen.blit(text, (panel_x + 10, y_pos))
                else:
                    text = small_font.render("Select an agent (CTRL+LMB)", True, (150, 150, 150))
                    screen.blit(text, (panel_x + 10, content_y))
            
            elif phylogenetic_tab == 'all':
                # Tüm ağaç sekmesi - gerçek ağaç görselleştirmesi
                full_tree = world.build_full_phylogenetic_tree()
                
                if full_tree and full_tree['by_generation']:
                    # Zoom'a göre node spacing hesapla
                    base_spacing_x = 120
                    base_spacing_y = 80
                    zoomed_spacing_x = int(base_spacing_x * phylogenetic_tree_zoom)
                    zoomed_spacing_y = int(base_spacing_y * phylogenetic_tree_zoom)
                    
                    # Ağaç layout'unu hesapla (geniş bir alan için, zoom'a göre)
                    tree_layout = world.calculate_tree_layout(full_tree, max(2000, panel_w * 3), 
                                                              node_spacing_x=zoomed_spacing_x, 
                                                              node_spacing_y=zoomed_spacing_y)
                    
                    if tree_layout:
                        # Scroll hesaplama (dikey ve yatay, zoom'a göre)
                        max_y = max(node['y'] for node in tree_layout.values()) if tree_layout else 0
                        max_x = max(node['x'] for node in tree_layout.values()) if tree_layout else 0
                        total_tree_height = max_y + 80
                        total_tree_width = max_x + 40
                        max_scroll = max(0, total_tree_height - content_h)
                        max_scroll_x = max(0, total_tree_width - (panel_w - 20))
                        phylogenetic_tree_scroll = max(0, min(phylogenetic_tree_scroll, max_scroll))
                        phylogenetic_tree_scroll_x = max(0, min(phylogenetic_tree_scroll_x, max_scroll_x))
                        
                        # Ağaç çizim alanı (clip)
                        tree_surface = pygame.Surface((panel_w - 20, content_h))
                        tree_surface.fill((15, 15, 30))
                        
                        # Bağlantıları çiz (L şeklinde - parent'tan aşağı, sonra yatay, sonra child'a)
                        for agent_id, node in tree_layout.items():
                            parent_id = node['parent']
                            if parent_id and parent_id in tree_layout:
                                parent_node = tree_layout[parent_id]
                                
                                # Pozisyonları hesapla (scroll ve zoom ile - hem dikey hem yatay)
                                # Zoom için merkez noktası (panel ortası)
                                center_x = (panel_w - 20) // 2
                                center_y = content_h // 2
                                
                                child_x = int((node['x'] - phylogenetic_tree_scroll_x - center_x) * phylogenetic_tree_zoom + center_x)
                                child_y = int((node['y'] - phylogenetic_tree_scroll + 20 - center_y) * phylogenetic_tree_zoom + center_y)
                                parent_x = int((parent_node['x'] - phylogenetic_tree_scroll_x - center_x) * phylogenetic_tree_zoom + center_x)
                                parent_y = int((parent_node['y'] - phylogenetic_tree_scroll + 20 - center_y) * phylogenetic_tree_zoom + center_y)
                                
                                # Sadece görünür alan içindeyse çiz
                                if (child_y >= -30 and child_y <= content_h + 30) or (parent_y >= -30 and parent_y <= content_h + 30):
                                    # Ölü ajanlar için gri, yaşayanlar için mavi
                                    line_color = (120, 120, 120) if not node['alive'] else (100, 150, 220)
                                    
                                    # L şeklinde çizgi: parent'tan aşağı, sonra yatay, sonra child'a
                                    mid_y = parent_y + (child_y - parent_y) // 2
                                    
                                    # Parent'tan aşağı (dikey)
                                    pygame.draw.line(tree_surface, line_color, 
                                                    (parent_x, parent_y), 
                                                    (parent_x, mid_y), 2)
                                    
                                    # Yatay çizgi
                                    pygame.draw.line(tree_surface, line_color, 
                                                    (parent_x, mid_y), 
                                                    (child_x, mid_y), 2)
                                    
                                    # Child'a (dikey)
                                    pygame.draw.line(tree_surface, line_color, 
                                                    (child_x, mid_y), 
                                                    (child_x, child_y), 2)
                        
                        # Node'ları çiz
                        center_x = (panel_w - 20) // 2
                        center_y = content_h // 2
                        for agent_id, node in tree_layout.items():
                            # Zoom için merkez noktası
                            x = int((node['x'] - phylogenetic_tree_scroll_x - center_x) * phylogenetic_tree_zoom + center_x)
                            y = int((node['y'] - phylogenetic_tree_scroll + 20 - center_y) * phylogenetic_tree_zoom + center_y)
                            
                            # Sadece görünür alan içindeyse çiz
                            if y >= -30 and y <= content_h + 30:
                                # Node rengi (yaşayan = yeşil, ölü = kırmızı)
                                if node['alive']:
                                    node_color = (100, 200, 100)
                                    text_color = (200, 255, 200)
                                else:
                                    node_color = (150, 80, 80)
                                    text_color = (255, 150, 150)
                                
                                # Node çemberi (zoom'a göre boyut)
                                node_radius = int(15 * phylogenetic_tree_zoom)
                                node_radius = max(5, min(30, node_radius))  # Min 5, max 30
                                pygame.draw.circle(tree_surface, node_color, (x, y), node_radius)
                                pygame.draw.circle(tree_surface, (255, 255, 255), (x, y), node_radius, 2)
                                
                                # ID yazısı
                                id_text = small_font.render(f"{agent_id}", True, text_color)
                                text_rect = id_text.get_rect(center=(x, y))
                                tree_surface.blit(id_text, text_rect)
                                
                                # Generation bilgisi (üstte)
                                gen_text = tiny_font.render(f"G{node['generation']}", True, (200, 200, 200))
                                gen_rect = gen_text.get_rect(center=(x, y - 25))
                                tree_surface.blit(gen_text, gen_rect)
                                
                                # Ölü ajanlar için "DEAD" yazısı
                                if not node['alive']:
                                    dead_text = tiny_font.render("DEAD", True, (255, 100, 100))
                                    dead_rect = dead_text.get_rect(center=(x, y + 25))
                                    tree_surface.blit(dead_text, dead_rect)
                        
                        # Tree surface'i ekrana çiz
                        screen.blit(tree_surface, (panel_x + 10, content_y))
                        
                        # Dikey scrollbar (sağda)
                        if max_scroll > 0:
                            scrollbar_w = 8
                            scrollbar_x = panel_x + panel_w - scrollbar_w - 5
                            scrollbar_h = content_h
                            scrollbar_y = content_y
                            
                            # Scrollbar arka planı
                            pygame.draw.rect(screen, (40, 40, 60), (scrollbar_x, scrollbar_y, scrollbar_w, scrollbar_h))
                            
                            # Scrollbar thumb
                            thumb_h = max(20, int(content_h * (content_h / total_tree_height)))
                            thumb_y = scrollbar_y + int((phylogenetic_tree_scroll / max_scroll) * (scrollbar_h - thumb_h)) if max_scroll > 0 else scrollbar_y
                            pygame.draw.rect(screen, (150, 150, 200), (scrollbar_x, thumb_y, scrollbar_w, thumb_h))
                        
                        # Yatay scrollbar (altta)
                        if max_scroll_x > 0:
                            h_scrollbar_h = 8
                            h_scrollbar_x = panel_x + 10
                            h_scrollbar_y = panel_y + panel_h - h_scrollbar_h - 15
                            h_scrollbar_w = panel_w - 20 - (scrollbar_w + 5 if max_scroll > 0 else 0)
                            
                            # Scrollbar arka planı
                            pygame.draw.rect(screen, (40, 40, 60), (h_scrollbar_x, h_scrollbar_y, h_scrollbar_w, h_scrollbar_h))
                            
                            # Scrollbar thumb
                            thumb_w = max(20, int(h_scrollbar_w * (h_scrollbar_w / total_tree_width)))
                            thumb_x = h_scrollbar_x + int((phylogenetic_tree_scroll_x / max_scroll_x) * (h_scrollbar_w - thumb_w)) if max_scroll_x > 0 else h_scrollbar_x
                            pygame.draw.rect(screen, (150, 150, 200), (thumb_x, h_scrollbar_y, thumb_w, h_scrollbar_h))
                    else:
                        text = small_font.render("Calculating tree layout...", True, (150, 150, 150))
                        screen.blit(text, (panel_x + 10, content_y))
                else:
                    text = small_font.render("No phylogenetic data available", True, (150, 150, 150))
                    screen.blit(text, (panel_x + 10, content_y))
            
            # Kapatma talimatı ve zoom bilgisi
            if phylogenetic_tab == 'all':
                zoom_text = small_font.render(f"Zoom: {phylogenetic_tree_zoom:.2f}x | CTRL+Wheel: Zoom | SHIFT+Wheel: Horizontal | Wheel: Vertical", True, (150, 150, 150))
                screen.blit(zoom_text, (panel_x + 10, panel_y + panel_h - 20))
            else:
                close_text = small_font.render("Press T to close | Click tabs to switch", True, (150, 150, 150))
                screen.blit(close_text, (panel_x + 10, panel_y + panel_h - 20))
        
        # Grafikler paneli
        if show_graphs:
            panel_x = SCREEN_WIDTH - 320
            panel_y = 10
            panel_w = 310
            panel_h = min(300, SCREEN_HEIGHT - 100)
            
            panel_surface = pygame.Surface((panel_w, panel_h))
            panel_surface.set_alpha(220)
            panel_surface.fill((20, 40, 20))
            screen.blit(panel_surface, (panel_x, panel_y))
            pygame.draw.rect(screen, (100, 150, 100), (panel_x, panel_y, panel_w, panel_h), 2)
            
            title = small_font.render("GRAPHS & CHARTS", True, (255, 255, 255))
            screen.blit(title, (panel_x + 10, panel_y + 10))
            
            y_pos = panel_y + 35
            if len(world.agents) > 0:
                # Genetik çeşitlilik grafiği (basit bar)
                text = small_font.render(f"Genetic Diversity: {world.hw_expected_heterozygosity:.4f}", True, (200, 255, 200))
                screen.blit(text, (panel_x + 10, y_pos))
                y_pos += 25
                
                # Fitness dağılımı
                if len(world.fitness_distribution) > 0:
                    avg_fitness = np.mean(world.fitness_distribution)
                    max_fitness = np.max(world.fitness_distribution)
                    min_fitness = np.min(world.fitness_distribution)
                    text = small_font.render(f"Fitness - Avg: {avg_fitness:.3f}", True, (200, 255, 200))
                    screen.blit(text, (panel_x + 10, y_pos))
                    y_pos += 20
                    text = small_font.render(f"Max: {max_fitness:.3f} Min: {min_fitness:.3f}", True, (200, 255, 200))
                    screen.blit(text, (panel_x + 10, y_pos))
                    y_pos += 25
                
                # Seleksiyon katsayıları
                if world.selection_coefficients:
                    text = small_font.render("Selection Coefficients:", True, (200, 255, 200))
                    screen.blit(text, (panel_x + 10, y_pos))
                    y_pos += 20
                    for locus, s in list(world.selection_coefficients.items())[:5]:  # İlk 5
                        text = small_font.render(f"  Locus {locus}: {s:.4f}", True, (180, 255, 180))
                        screen.blit(text, (panel_x + 10, y_pos))
                        y_pos += 18
                
                # Genetik mesafe
                if 'mean' in world.genetic_distances:
                    text = small_font.render(f"Genetic Distance: {world.genetic_distances['mean']:.4f}", True, (200, 255, 200))
                    screen.blit(text, (panel_x + 10, y_pos))
                    y_pos += 25
            
            close_text = small_font.render("Press G to close", True, (150, 150, 150))
            screen.blit(close_text, (panel_x + 10, panel_y + panel_h - 20))
        
        # Filtreleme paneli
        if show_filters:
            panel_x = SCREEN_WIDTH // 2 - 200
            panel_y = 50
            panel_w = 400
            panel_h = min(400, SCREEN_HEIGHT - 100)
            
            panel_surface = pygame.Surface((panel_w, panel_h))
            panel_surface.set_alpha(220)
            panel_surface.fill((40, 20, 20))
            screen.blit(panel_surface, (panel_x, panel_y))
            pygame.draw.rect(screen, (150, 100, 100), (panel_x, panel_y, panel_w, panel_h), 2)
            
            title = small_font.render("FILTER AGENTS", True, (255, 255, 255))
            screen.blit(title, (panel_x + panel_w // 2 - title.get_width() // 2, panel_y + 10))
            
            y_pos = panel_y + 40
            if len(world.agents) > 0:
                # Filtrelenmiş ajan sayısı
                filtered = world.filter_agents(**filter_params) if filter_params else world.agents
                text = small_font.render(f"Total Agents: {len(world.agents)} | Filtered: {len(filtered)}", True, (255, 200, 200))
                screen.blit(text, (panel_x + 20, y_pos))
                y_pos += 30
                
                # Filtre bilgileri
                text = small_font.render("Current Filters:", True, (255, 255, 255))
                screen.blit(text, (panel_x + 20, y_pos))
                y_pos += 25
                
                if filter_params:
                    for key, value in filter_params.items():
                        text = small_font.render(f"  {key}: {value}", True, (255, 200, 200))
                        screen.blit(text, (panel_x + 20, y_pos))
                        y_pos += 20
                else:
                    text = small_font.render("  No filters applied", True, (150, 150, 150))
                    screen.blit(text, (panel_x + 20, y_pos))
                    y_pos += 25
                
                # Yaş grupları
                age_dist = world.get_age_distribution()
                text = small_font.render(f"Age Groups - J:{age_dist['juvenile']} A:{age_dist['adult']} S:{age_dist['senescent']}", True, (255, 200, 200))
                screen.blit(text, (panel_x + 20, y_pos))
                y_pos += 25
                
                # Bölgeler
                if world.regions:
                    text = small_font.render("Regions:", True, (255, 255, 255))
                    screen.blit(text, (panel_x + 20, y_pos))
                    y_pos += 20
                    for region_id, agent_ids in list(world.regions.items())[:4]:
                        text = small_font.render(f"  Region {region_id}: {len(agent_ids)} agents", True, (255, 200, 200))
                        screen.blit(text, (panel_x + 20, y_pos))
                        y_pos += 18
            
            close_text = small_font.render("Press F to close", True, (150, 150, 150))
            screen.blit(close_text, (panel_x + panel_w // 2 - close_text.get_width() // 2, panel_y + panel_h - 20))
        
        # Yaş grupları dağılımı (stats panelinde)
        if show_stats and len(world.agents) > 0:
            age_dist = world.get_age_distribution()
            age_text = small_font.render(f"Age Groups - J:{age_dist['juvenile']} A:{age_dist['adult']} S:{age_dist['senescent']}", True, (150, 150, 150))
            screen.blit(age_text, (10, SCREEN_HEIGHT - 80))
        
        # İstatistiksel testler paneli (I tuşu)
        if show_statistical_tests and len(world.agents) > 0:
            panel_w = 500
            panel_h = 600
            panel_x = SCREEN_WIDTH // 2 - panel_w // 2
            panel_y = 50
            
            # Panel arka planı
            panel_surface = pygame.Surface((panel_w, panel_h))
            panel_surface.set_alpha(240)
            panel_surface.fill((20, 20, 40))
            screen.blit(panel_surface, (panel_x, panel_y))
            pygame.draw.rect(screen, (100, 100, 150), (panel_x, panel_y, panel_w, panel_h), 2)
            
            title = font.render("STATISTICAL TESTS", True, (255, 255, 255))
            screen.blit(title, (panel_x + 10, panel_y + 10))
            
            y_pos = panel_y + 40
            line_height = 22
            
            # Hardy-Weinberg testi
            hw_result = world.test_hardy_weinberg(locus=0)
            if hw_result:
                text = small_font.render("=== HARDY-WEINBERG TEST ===", True, (200, 200, 255))
                screen.blit(text, (panel_x + 10, y_pos))
                y_pos += line_height
                
                text = small_font.render(f"Chi-square: {hw_result['chi_square']:.4f}", True, (255, 255, 255))
                screen.blit(text, (panel_x + 10, y_pos))
                y_pos += line_height
                
                text = small_font.render(f"p-value: {hw_result['p_value']:.4f}", True, (255, 255, 255))
                screen.blit(text, (panel_x + 10, y_pos))
                y_pos += line_height
                
                status = "REJECT H0" if hw_result['reject_H0'] else "ACCEPT H0"
                color = (255, 100, 100) if hw_result['reject_H0'] else (100, 255, 100)
                text = small_font.render(f"Result: {status}", True, color)
                screen.blit(text, (panel_x + 10, y_pos))
                y_pos += line_height * 2
            
            # Effective population size
            Ne = world.calculate_effective_population_size()
            text = small_font.render(f"=== EFFECTIVE POPULATION SIZE ===", True, (200, 200, 255))
            screen.blit(text, (panel_x + 10, y_pos))
            y_pos += line_height
            text = small_font.render(f"Ne: {Ne:.2f} (N={len(world.agents)})", True, (255, 255, 255))
            screen.blit(text, (panel_x + 10, y_pos))
            y_pos += line_height * 2
            
            # Trend analizi
            if len(world.genetic_diversity_history) > 5:
                trend_result = world.test_trend(list(world.genetic_diversity_history), test_type='kendall')
                if trend_result:
                    text = small_font.render("=== GENETIC DIVERSITY TREND ===", True, (200, 200, 255))
                    screen.blit(text, (panel_x + 10, y_pos))
                    y_pos += line_height
                    text = small_font.render(f"Kendall's tau: {trend_result['tau']:.4f}", True, (255, 255, 255))
                    screen.blit(text, (panel_x + 10, y_pos))
                    y_pos += line_height
                    text = small_font.render(f"p-value: {trend_result['p_value']:.4f}", True, (255, 255, 255))
                    screen.blit(text, (panel_x + 10, y_pos))
                    y_pos += line_height
                    trend_text = f"Trend: {trend_result['trend']}"
                    if trend_result['significant']:
                        trend_text += " (SIGNIFICANT)"
                    text = small_font.render(trend_text, True, (255, 255, 100) if trend_result['significant'] else (255, 255, 255))
                    screen.blit(text, (panel_x + 10, y_pos))
                    y_pos += line_height * 2
            
            # Türleşme analizi
            speciation_result = world.detect_speciation_event(threshold=0.3)
            if speciation_result:
                text = small_font.render("=== SPECIATION ANALYSIS ===", True, (200, 200, 255))
                screen.blit(text, (panel_x + 10, y_pos))
                y_pos += line_height
                if speciation_result['speciation_detected']:
                    text = small_font.render("SPECIATION DETECTED!", True, (255, 100, 100))
                    screen.blit(text, (panel_x + 10, y_pos))
                    y_pos += line_height
                    text = small_font.render(f"Fst: {speciation_result['fst']:.4f}", True, (255, 255, 255))
                    screen.blit(text, (panel_x + 10, y_pos))
                    y_pos += line_height
                    if speciation_result['regions']:
                        text = small_font.render(f"Regions: {speciation_result['regions']}", True, (255, 255, 255))
                        screen.blit(text, (panel_x + 10, y_pos))
                else:
                    text = small_font.render(f"No speciation (Fst: {speciation_result['fst']:.4f})", True, (150, 150, 150))
                    screen.blit(text, (panel_x + 10, y_pos))
                y_pos += line_height * 2
            
            # Kapatma talimatı
            text = small_font.render("Press 'I' to close", True, (150, 150, 150))
            screen.blit(text, (panel_x + 10, panel_y + panel_h - 25))
        
        # Fitness Landscape görselleştirmesi (L tuşu)
        if show_fitness_landscape and len(world.agents) > 5:
            landscape_data = world.calculate_fitness_landscape(resolution=30)
            if landscape_data:
                # Basit 2D heatmap görselleştirmesi
                panel_w = 400
                panel_h = 400
                panel_x = SCREEN_WIDTH - panel_w - 10
                panel_y = 10
                
                # Panel arka planı
                panel_surface = pygame.Surface((panel_w, panel_h))
                panel_surface.set_alpha(240)
                panel_surface.fill((20, 20, 40))
                screen.blit(panel_surface, (panel_x, panel_y))
                pygame.draw.rect(screen, (100, 100, 150), (panel_x, panel_y, panel_w, panel_h), 2)
                
                title = small_font.render("FITNESS LANDSCAPE", True, (255, 255, 255))
                screen.blit(title, (panel_x + 10, panel_y + 10))
                
                # Heatmap çiz
                landscape = landscape_data['landscape']
                max_fit = landscape_data['max_fitness']
                min_fit = landscape_data['min_fitness']
                range_fit = max_fit - min_fit if max_fit > min_fit else 1
                
                cell_w = (panel_w - 40) // landscape.shape[1]
                cell_h = (panel_h - 60) // landscape.shape[0]
                
                for i in range(landscape.shape[0]):
                    for j in range(landscape.shape[1]):
                        val = landscape[i, j]
                        # Normalize to 0-1
                        normalized = (val - min_fit) / range_fit
                        # Color gradient (blue to red)
                        r = int(255 * normalized)
                        g = int(100 * (1 - normalized))
                        b = int(255 * (1 - normalized))
                        
                        x = panel_x + 20 + j * cell_w
                        y = panel_y + 40 + i * cell_h
                        pygame.draw.rect(screen, (r, g, b), (x, y, cell_w, cell_h))
                
                # Peaks göster
                peaks = world.find_fitness_peaks(landscape_data, num_peaks=3)
                for peak in peaks:
                    px = panel_x + 20 + int(peak['x'] * (panel_w - 40))
                    py = panel_y + 40 + int(peak['y'] * (panel_h - 60))
                    pygame.draw.circle(screen, (255, 255, 0), (px, py), 5)
                
                # Bilgi
                info_y = panel_y + panel_h - 20
                text = small_font.render(f"Max: {max_fit:.2f} | Min: {min_fit:.2f} | Peaks: {len(peaks)}", True, (200, 200, 200))
                screen.blit(text, (panel_x + 10, info_y))
        
        # Deney tasarımı paneli (D tuşu)
        if show_experiment_design:
            panel_w = 500
            panel_h = 500
            panel_x = SCREEN_WIDTH // 2 - panel_w // 2
            panel_y = 50
            
            panel_surface = pygame.Surface((panel_w, panel_h))
            panel_surface.set_alpha(240)
            panel_surface.fill((20, 20, 40))
            screen.blit(panel_surface, (panel_x, panel_y))
            pygame.draw.rect(screen, (100, 100, 150), (panel_x, panel_y, panel_w, panel_h), 2)
            
            title = font.render("EXPERIMENT DESIGN", True, (255, 255, 255))
            screen.blit(title, (panel_x + 10, panel_y + 10))
            
            y_pos = panel_y + 50
            line_height = 20
            
            # Sensitivity analysis bilgisi
            text = small_font.render("=== SENSITIVITY ANALYSIS ===", True, (200, 200, 255))
            screen.blit(text, (panel_x + 10, y_pos))
            y_pos += line_height * 2
            
            # Effective population size
            Ne = world.calculate_effective_population_size()
            text = small_font.render(f"Effective Population Size (Ne): {Ne:.2f}", True, (255, 255, 255))
            screen.blit(text, (panel_x + 10, y_pos))
            y_pos += line_height * 2
            
            # Phase transitions
            if len(world.population_history) > 40:
                transitions = world.detect_phase_transitions(list(world.population_history), window_size=10)
                text = small_font.render(f"=== PHASE TRANSITIONS ===", True, (200, 200, 255))
                screen.blit(text, (panel_x + 10, y_pos))
                y_pos += line_height
                text = small_font.render(f"Detected: {len(transitions)} transitions", True, (255, 255, 255))
                screen.blit(text, (panel_x + 10, y_pos))
                y_pos += line_height
                if transitions:
                    latest = transitions[-1]
                    text = small_font.render(f"Latest: Tick {latest['time']}, Change: {latest['change']:.3f}", True, (255, 255, 100))
                    screen.blit(text, (panel_x + 10, y_pos))
                    y_pos += line_height * 2
            
            # Population forecast
            forecast = world.forecast_population(horizon=20, method='linear')
            if forecast:
                text = small_font.render("=== POPULATION FORECAST ===", True, (200, 200, 255))
                screen.blit(text, (panel_x + 10, y_pos))
                y_pos += line_height
                text = small_font.render(f"Current: {forecast['current']}", True, (255, 255, 255))
                screen.blit(text, (panel_x + 10, y_pos))
                y_pos += line_height
                if forecast['predicted_final']:
                    text = small_font.render(f"Predicted (20 ticks): {forecast['predicted_final']:.1f}", True, (100, 255, 100))
                    screen.blit(text, (panel_x + 10, y_pos))
                    y_pos += line_height * 2
            
            text = small_font.render("Press 'D' to close", True, (150, 150, 150))
            screen.blit(text, (panel_x + 10, panel_y + panel_h - 25))
        
        # Network analizi paneli (N tuşu)
        if show_network_analysis:
            panel_w = 500
            panel_h = 500
            panel_x = SCREEN_WIDTH // 2 - panel_w // 2
            panel_y = 50
            
            panel_surface = pygame.Surface((panel_w, panel_h))
            panel_surface.set_alpha(240)
            panel_surface.fill((20, 20, 40))
            screen.blit(panel_surface, (panel_x, panel_y))
            pygame.draw.rect(screen, (100, 100, 150), (panel_x, panel_y, panel_w, panel_h), 2)
            
            title = font.render("NETWORK ANALYSIS", True, (255, 255, 255))
            screen.blit(title, (panel_x + 10, panel_y + 10))
            
            y_pos = panel_y + 50
            line_height = 20
            
            # Migration network
            migration_net = world.build_migration_network(distance_threshold=50)
            if migration_net:
                text = small_font.render("=== MIGRATION NETWORK ===", True, (200, 200, 255))
                screen.blit(text, (panel_x + 10, y_pos))
                y_pos += line_height
                
                metrics = world.calculate_network_metrics(migration_net)
                if metrics:
                    text = small_font.render(f"Nodes: {metrics['num_nodes']} | Edges: {metrics['num_edges']}", True, (255, 255, 255))
                    screen.blit(text, (panel_x + 10, y_pos))
                    y_pos += line_height
                    text = small_font.render(f"Avg Degree: {metrics['avg_degree']:.2f} | Density: {metrics['density']:.3f}", True, (255, 255, 255))
                    screen.blit(text, (panel_x + 10, y_pos))
                    y_pos += line_height * 2
            
            # Social network
            social_net = world.build_social_network()
            if social_net:
                text = small_font.render("=== SOCIAL NETWORK ===", True, (200, 200, 255))
                screen.blit(text, (panel_x + 10, y_pos))
                y_pos += line_height
                
                metrics = world.calculate_network_metrics(social_net)
                if metrics:
                    text = small_font.render(f"Nodes: {metrics['num_nodes']} | Edges: {metrics['num_edges']}", True, (255, 255, 255))
                    screen.blit(text, (panel_x + 10, y_pos))
                    y_pos += line_height
                    text = small_font.render(f"Avg Degree: {metrics['avg_degree']:.2f} | Density: {metrics['density']:.3f}", True, (255, 255, 255))
                    screen.blit(text, (panel_x + 10, y_pos))
                    y_pos += line_height * 2
            
            text = small_font.render("Press 'N' to close", True, (150, 150, 150))
            screen.blit(text, (panel_x + 10, panel_y + panel_h - 25))
        
        # Kısayol tuşları bilgisi
        if not show_stats and not show_statistical_tests and not show_fitness_landscape and not show_experiment_design and not show_network_analysis:
            shortcuts = [
                "T: Phylogenetic | G: Graphs | F: Filters | E: Export | I: Stats Tests",
                "L: Fitness Landscape | D: Experiment Design | N: Network | S: Statistics"
            ]
            for i, shortcut in enumerate(shortcuts):
                text = small_font.render(shortcut, True, (100, 100, 100))
                screen.blit(text, (10, SCREEN_HEIGHT - 100 + i * 15))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()


if __name__ == "__main__":
    main()

