import numpy as np
import matplotlib.pyplot as plt
import imageio
import random
from scipy.special import erfcinv
from pathlib import Path
from core.elements import *
import random


def db2lin(db):
    #print(db, 10 ** (db / 10))
    return 10 ** (db / 10)


class TrafficMatrixSimulator:
    def __init__(self, nodes, M, transceiver_strategy):
        self.nodes = nodes
        self.transceiver_strategy = transceiver_strategy
        self.traffic_matrix = self.create_traffic_matrix(M)
        self.frames = []
        self._weighted_paths = self.generate_weighted_paths()

    def create_traffic_matrix(self, M):
        num_nodes = len(self.nodes)
        traffic_matrix = np.full((num_nodes, num_nodes), 100 * M, dtype=float)

        # 🔥 Assicuriamoci che ogni connessione abbia almeno un po' di traffico 🔥
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    traffic_matrix[i, j] = max(traffic_matrix[i, j], random.uniform(10, 100))

        np.fill_diagonal(traffic_matrix, 0)
        return traffic_matrix

    def generate_weighted_paths(self):
        return {f"{i}->{j}": random.uniform(25, 40) for i in self.nodes for j in self.nodes if i != j}

    def choose_random_connection(self):
        indices = [(i, j) for i in range(len(self.nodes)) for j in range(len(self.nodes))
                   if i != j and self.traffic_matrix[i, j] > 0]  # 🔥 Solo connessioni attive!
        if not indices:
            return None, None
        return random.choice(indices)

    def calculate_bit_rate_l(self, source, destination):
        path = f"{source}->{destination}"
        Rs = 32e9
        Bn = 12.5e9
        BERt = 1e-3

        th_100 = 2 * (erfcinv(2 * BERt)) ** 2 * Rs / Bn
        th_200 = (14 / 3) * (erfcinv(BERt * 3 / 2)) ** 2 * Rs / Bn
        th_400 = 10 * (erfcinv(BERt * 8 / 3)) ** 2 * Rs / Bn
        #print(th_100, th_200, th_400)

        GSNR = db2lin(self._weighted_paths.get(path, 15))
        #print(self._weighted_paths.get(path, 15))
        #print(GSNR)

        if self.transceiver_strategy == 'fixed-rate':
            if GSNR >= th_100:
                return 100 * 1e9
            return 0
        elif self.transceiver_strategy == 'flex-rate':
            if GSNR < th_100:
                return 0
            elif th_100 <= GSNR < th_200:
                return 100 * 1e9
            elif th_200 <= GSNR < th_400:
                return 200 * 1e9
            elif GSNR >= th_400:
                return 400 * 1e9
        elif self.transceiver_strategy == 'shannon':
            return 2 * Rs * np.log2(1 + GSNR * Rs / Bn)
        return 0

    def update_traffic_matrix(self, source, destination):
        bit_rate = self.calculate_bit_rate_l(source, destination)
        if bit_rate > 0:
        #print(self.traffic_matrix[source][destination], bit_rate * 10 ** (-9))
            self.traffic_matrix[source][destination] -= bit_rate*10**(-9)
            #print(self.traffic_matrix[source][destination])
            if self.traffic_matrix[source][destination] <= 0:
                self.traffic_matrix[source][destination] = 0
            return True
        else:
            print("mino")
        return False

    def check_traffic_matrix(self):
        return np.any(self.traffic_matrix > 0)

    def save_frame(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        cax = ax.imshow(self.traffic_matrix, cmap='coolwarm', interpolation='nearest')
        fig.colorbar(cax, label='Traffic Load')

        num_nodes = self.traffic_matrix.shape[0]
        for i in range(num_nodes):
            for j in range(num_nodes):
                ax.text(j, i, f'{self.traffic_matrix[i, j]:.1f}', ha='center', va='center', color='black')

        plt.title(f"Traffic Matrix Evolution - {self.transceiver_strategy}")
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())
        self.frames.append(image)
        plt.close(fig)

    def simulate(self):
        max_iterations = 1500
        iterations = 0
        used_connections = set()  # 🔥 Per tracciare le connessioni scelte!

        while self.check_traffic_matrix() and iterations < max_iterations:
            self.save_frame()
            source, destination = self.choose_random_connection()
            if source is not None and destination is not None:
                used_connections.add((source, destination))  # ✅ Registra la connessione usata
                self.update_traffic_matrix(source, destination)
                #if not self.update_traffic_matrix(source, destination):
            iterations += 1
        self.save_frame()

        # 🔥 Stampiamo le connessioni mai scelte
        all_connections = {(i, j) for i in range(len(self.nodes)) for j in range(len(self.nodes)) if i != j}
        unused_connections = all_connections - used_connections
        print(f"Connessioni mai usate: {unused_connections}")  # 🔎 Debug

    def create_gif(self, filename):
        imageio.mimsave(filename, self.frames, duration=0.5)
        print(f"GIF saved as {filename}")


nodes = {i: f'Node {i}' for i in range(5)}
for strategy in ['fixed-rate', 'flex-rate', 'shannon']:
    simulator = TrafficMatrixSimulator(nodes, M=35, transceiver_strategy=strategy)
    simulator.simulate()
    simulator.create_gif(filename=f'traffic_matrix_{strategy}_changeddd_35.gif')
    print(f"{strategy} is done")