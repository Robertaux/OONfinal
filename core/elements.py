import json
from core.parameters import *
import matplotlib.pyplot as plt
import math
from core.utils import *
import numpy as np
from scipy.special import erfcinv
from core.math_utils import *
import random
import os
import pandas as pd

class Signal_information(object):
    def __init__(self, signal_power, path, channel=None):
        self._signal_power=signal_power
        self._noise_power=0.0
        self._latency=0.0
        self._path=path
        self._channel = channel

    @property
    def signal_power(self):
        return self._signal_power

    @signal_power.setter
    def signal_power(self, sign):
        self._signal_power=sign

    def update_signal_power(self, increment_s):
        self._signal_power+=increment_s

    @property
    def noise_power(self):
        return self._noise_power

    @noise_power.setter
    def noise_power(self, nois):
        self._noise_power=nois

    def update_noise_power(self, increment_n):
        self._noise_power+=increment_n

    @property
    def latency(self):
        return self._latency

    @latency.setter
    def latency(self, val_lat):
        self._latency=val_lat

    def update_latency(self, increment_l):
        self._latency+=increment_l

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path_n):
        self._path=path_n

    def update_path(self, node_n):
        self._path.append(node_n)

    def update_path_c(self):
        self._path.pop(0)

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value):
        self._channel = value

class Lightpath(Signal_information):
    def __init__(self, signal_power, path, channel):
        super().__init__(signal_power, path)
        self._channel = channel
        self._Rs = 32e9
        self._df = 50e9
        self._isnr= 0.0

    @property
    def Rs(self):
        return self._Rs

    @property
    def df(self):
        return self._df

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, channel_value):
        self._channel=channel_value

    def noise_power_l(self, noisepower):
        self._noise_power=noisepower

    def latency_l(self, latency):
        self._latency=latency

    @property
    def isnr(self):
        return self._channel

    def isnr_l(self, line):
        isnr=line.compute_gain()
        self._isnr += isnr
        return isnr

    def final_gsnr(self, line):
        gsnr_db = 10 * math.log10(1 / self.isnr_l(line))
        gsnr_linear = math.pow(10, gsnr_db / 10)
        return gsnr_linear

class Node(object):
    def __init__(self, node_data):
        self._label=node_data.get("label", "")
        self._position=node_data.get("position", (0.0, 0.0))
        self._connected_nodes=node_data.get("connected_nodes", [])
        self._successive={}
        self._successiven={}
        self._switching_matrix = node_data.get("switching_matrix", {})
        self._transceiver = node_data.get("transceiver", "fixed-rate")

    @property
    def label(self):
        return self._label

    @property
    def position(self):
        return self._position

    @property
    def connected_nodes(self):
        return self._connected_nodes

    @property
    def successive(self):
        return self._successive

    @successive.setter
    def successive(self, val_s):
        self._successive=val_s

    @property
    def successiven(self):
        return self._successiven

    @successiven.setter
    def successiven(self, val_s):
        self._successiven = val_s

    @property
    def switching_matrix(self):
        return self._switching_matrix

    @switching_matrix.setter
    def switching_matrix(self, value):
        self._switching_matrix = value

    @property
    def transceiver(self):
        return self._transceiver

    @transceiver.setter
    def transceiver(self, value):
        self._transceiver = value

    def propagate(self, signal_info):
        signal_info.update_path_c()
        if signal_info.path:
            line_label=self.label+signal_info.path[0]
            if line_label in self._successive:
                self._successive[line_label].propagate(signal_info, signal_info.channel)

    def propagate_lightpath(self, lightpath):
        lightpath.update_path_c()
        if lightpath.path:
            line_label=self.label+lightpath.path[0]
            if line_label in self._successive:
                line_obj=self._successive[line_label]
                optimal_power=line_obj.optimized_launch_power(lightpath)
                lightpath.signal_power=optimal_power
                self._successive[line_label].propagate_lightpath(lightpath)

    def probe(self, signal_info):
        signal_info.update_path_c()
        if signal_info.path:
            line_label=self.label+signal_info.path[0]
            if line_label in self._successive:
                self._successive[line_label].probe(signal_info)

class Line(object):
    def __init__(self, label, length, channels_numb=10):
        self._label=label
        self._length=length
        self._successive={}
        self._state=[1 for _ in range(channels_numb)]
        self._channels_numb=channels_numb
        self._n_amplifiers = math.ceil(length / 80000)
        self._gain_dB = 16
        self._noise_figure_dB = 5.5
        self._gain = 10 ** (self._gain_dB / 10)
        self._noise_figure = 10 ** (self._noise_figure_dB / 10)
        self._alpha_db = 0.2e-3
        self._beta2_abs = 2.13e-26
        self._gamma = 1.27e-3
        self._alpha = self._alpha_db / (20 * math.log10(math.e))
        self._distance=80000
        self._Bn = 12.5e9

    @property
    def label(self):
        return self._label

    @property
    def length(self):
        return self._length

    @property
    def successive(self):
        return self._successive

    @successive.setter
    def successive(self, succ):
        self._successive=succ

    @property
    def successiven(self):
        return self._successive

    @successiven.setter
    def successiven(self, succ):
        self._successive=succ

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        for i in range(len(self.state)):
            self.state[i] = value

    @property
    def n_amplifiers(self):
        return self._n_amplifiers

    @property
    def gain(self):
        return self._gain

    @property
    def noise_figure(self):
        return self._noise_figure

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta2_abs(self):
        return self._beta2_abs

    @property
    def gamma(self):
        return self._gamma

    @property
    def distance(self):
        return self._distance

    @property
    def Bn(self):
        return self._Bn

    @property
    def get_channels_numb(self):
        return self._channels_numb

    def ase_generation(self):
        frequency = 193.414e12
        Bn = 12.5e9
        ase = self._n_amplifiers * Planck * frequency * Bn * self._noise_figure * (self._gain - 1)
        return ase

    def nli_generation(self, lightpath):
        PI = np.pi
        Nspan = self._n_amplifiers - 1
        Bn = 12.5e9
        Leff = (1 - np.exp(-2 * self._alpha * self._distance)) / (2 * self._alpha)
        eta_nli = (16 / (27 * PI)) * np.log( ((PI ** 2) * self._beta2_abs * (lightpath.Rs ** 2) * self._channels_numb ** (2 * (lightpath.Rs) / lightpath.df)) / (2 * self._alpha)) * ((self._alpha * (self._gamma * Leff) ** 2) / (self._beta2_abs * lightpath.Rs ** 3))
        logdeln=np.log( ((PI ** 2) * self._beta2_abs * (lightpath.Rs ** 2) * self._channels_numb ** (2 * (lightpath.Rs) / lightpath.df)) / (2 * self._alpha))
        nli = (lightpath.signal_power) ** 3 * eta_nli * Nspan * Bn
        return nli,eta_nli

    def crosstalk_generation(self, lightpath):
        alpha_xt = 1e-4
        crosstalk_power = lightpath.signal_power * (lightpath.Rs / lightpath.df) * alpha_xt
        if 0<lightpath.channel<9:
            if self._state[lightpath.channel-1]==1:
                if self._state[lightpath.channel+1] == 1:
                    return 2*crosstalk_power
                return crosstalk_power
            elif self._state[lightpath.channel+1]==1:
                return crosstalk_power
        elif lightpath.channel==0:
            if self._state[lightpath.channel + 1] == 1:
                return crosstalk_power
        else:
            if self._state[lightpath.channel - 1] == 1:
                return crosstalk_power
        return 0

    def optimized_launch_power(self, lightpath):
        ase=self.ase_generation()
        nli, eta_nli=self.nli_generation(lightpath)
        optimal_power = (ase / (2 * eta_nli * (self._n_amplifiers - 1) * self._Bn))
        return math.pow(optimal_power, 1/3)

    def compute_gain(self, lightpath):
        power_signal = lightpath.signal_power
        power_noise = lightpath.noise_power
        gsnr = power_signal / power_noise
        isnr = 1 / gsnr
        return isnr

    def get_state(self, channel):
        if 0 <= channel < len(self.state):
            return self.state[channel]
        else:
            raise ValueError("Channel must be an integer greater than 0.")

    def get_states(self):
        states=[]
        for channel in range(10):
            states.append(self.state[channel])
        return states

    def state_occupied(self, channel):
        if 0<=channel<len(self.state):
            self.state[channel]=0
            #print(f"Channel {channel} occupied.")

    def state_freed(self, channel):
        if 0<=channel<len(self.state):
            self.state[channel]=1

    def channels_freed(self):
        for i in range(len(self.state)):
            self.state[i]=1

    def latency_generation(self):
        speed=(2/3)*speed_light
        latency=self._length/speed
        return latency

    def noise_generation(self, signal_power):
        noise_power=1e-9*signal_power*self._length
        return noise_power

    def noise_generation_l(self, lightpath):
        ase = self.ase_generation()
        nli,eta_nli = self.nli_generation(lightpath)
        xt = self.crosstalk_generation(lightpath)
        noise_power = ase + nli + xt
        return noise_power

    def propagate_lightpath(self, lightpath):
        if 0<=lightpath.channel<len(self._state):
            lightpath.update_noise_power(self.noise_generation_l(lightpath))
            lightpath.update_latency(self.latency_generation())
            if lightpath.path and lightpath.path[0] in self._successive:
                self.state_occupied(lightpath.channel)
                self._successive[lightpath.path[0]].propagate_lightpath(lightpath)

    def probe(self, signal_info):
        signal_info.update_noise_power(self.noise_generation(signal_info.signal_power))
        signal_info.update_latency(self.latency_generation())
        if signal_info.path and signal_info.path[0] in self._successive:
            self._successive[signal_info.path[0]].probe(signal_info)

class Connection:
    def __init__(self, input_node, output_node, signal_power, channel=None):
        self.input=input_node
        self.output=output_node
        self.signal_power=signal_power
        self.channel=channel
        self.latency=0.0
        self.snr=0.0
        self.bit_rate = 0.0

    def set_channel(self, channel):
        self.channel = channel

    def free_channels(self, lines, channel):
        for line in lines:
            line.state_freed(channel)

    def occupy_channels(self, lines, channel):
        for line in lines:
            line.state_occupied(channel)

class Network:
    def __init__(self, json_file):
        self._nodes={}
        self._lines={}
        self._switching_matrices={}

        if json_file is not None:
            with open(json_file) as f:
                data=json.load(f)
                #print(data)
                for node_label, node_data in data.items():
                    self._nodes[node_label]=Node({
                        "label": node_label,
                        "connected_nodes": node_data["connected_nodes"],
                        "position": tuple(node_data["position"]),
                        "switching_matrix": node_data["switching_matrix"],
                        "transceiver": node_data.get("transceiver", "fixed-rate")
                    })
                    #print(node_label)
                    #print(node_data["connected_nodes"])
                    #print(node_data["position"])
                    #print(node_data["switching_matrix"])
                    #print(node_data.get("transceiver"))
                    self._switching_matrices[node_label]=node_data["switching_matrix"]
                for node_label, node in self._nodes.items():
                    for neighbor_label in node.connected_nodes:
                        line_label=node_label+neighbor_label
                        if line_label not in self._lines and neighbor_label in self._nodes:
                            line_length=math.dist(node.position, self._nodes[neighbor_label].position)
                            self._lines[line_label]=Line(line_label, line_length)
                self.connect()
        self._weighted_paths = None
        self.rs_latency = self.route_space()
        self.rs_snr = self.route_space()
        self.create_wp_dataframe(0.001)

    @property
    def nodes(self):
        return self._nodes

    @property
    def lines(self):
        return self._lines

    @property
    def weighted_paths(self):
        return self._weighted_paths

    def connect(self):
        for node in self.nodes.values():
            node.successiven = {label: self.nodes[label] for label in node.connected_nodes}
            node.successive = {line.label: line for line in self.lines.values() if node.label in line.label}
        for line in self.lines.values():
            node_labels=list(line.label)
            line.successive={node_labels[0]: self.nodes[node_labels[0]], node_labels[1]: self.nodes[node_labels[1]]}
        for node_label in self.nodes:
            self.initialize_sm(node_label)

    def initialize_sm(self, node_label):
        connected_nodes = self.nodes[node_label].connected_nodes
        switching_matrix = {
            neighbor: {
                n: np.zeros(10) if neighbor == n else np.ones(10)
                for n in connected_nodes
            }
            for neighbor in connected_nodes
        }
        self.nodes[node_label].switching_matrix = switching_matrix
        return switching_matrix

    def find_paths(self, source_label, destination_label):
        def search(current_node, path):
            if current_node.label==destination_label:
                all_paths.append(path.copy())
                return
            crossed.add(current_node.label)
            for neighbor_label, neighbor_node in current_node.successiven.items():
                if neighbor_label not in crossed:
                    path.append(neighbor_label)
                    search(neighbor_node, path)
                    path.pop()
            crossed.remove(current_node.label)
        crossed=set()
        all_paths=[]
        source_node=self.nodes[source_label]
        search(source_node, [source_label])
        return all_paths

    def propagate(self, signal_info):
        self._nodes[signal_info.path[0]].propagate(signal_info)
        return signal_info

    def propagate_lightpath(self, lightpath):
        self._nodes[lightpath.path[0]].propagate_lightpath(lightpath)
        return lightpath

    def draw(self):
        for node in self.nodes.values():
            plt.scatter(node.position[0], node.position[1], label=node.label)
        for line in self.lines.values():
            node1=line.label[0]
            node2=line.label[1]
            x1, y1=self.nodes[node1].position
            x2, y2=self.nodes[node2].position
            plt.plot([x1, x2], [y1, y2], linestyle='--', color='black')
            mid_x=(x1+x2)/2
            mid_y=(y1+y2)/2
            distance=round(line.length, 2)
            plt.text(mid_x, mid_y, str(distance), horizontalalignment='center', verticalalignment='center')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()

    def create_wp_dataframe(self, signal_power_watts):
        results=[]
        for source_label in self.nodes:
            for destination_label in self.nodes:
                if source_label!=destination_label:
                    paths=self.find_paths(source_label, destination_label)
                    for path in paths:
                        path_string="->".join(path)
                        signal_info=Signal_information(signal_power_watts, path)
                        self.probe(signal_info)
                        total_latency=signal_info.latency
                        total_noise_power=signal_info.noise_power
                        total_snr=signal_info.signal_power/total_noise_power
                        total_snr_db=lin2db(total_snr)
                        results.append([path_string, total_latency, total_noise_power, total_snr_db])
        columns = ["Path", "Total Latency (s)", "Total Noise Power (W)", "SNR (dB)"]
        self._weighted_paths = pd.DataFrame(results, columns=columns)

    def find_best_snr(self, source_label, dest_label, signal_power):
        best_path = None
        best_latency = float('inf')
        best_snr = float('-inf')
        best_channel=None
        paths = self.find_paths(source_label, dest_label)
        for path in paths:
            for channel in range(10):
                if self.is_channel_free_along_path(path, channel):
                    row= self._weighted_paths[self._weighted_paths["Path"] == "->".join(path)]
                    if not row.empty:
                        total_latency = row["Total Latency (s)"].values[0]
                        total_snr_db = row["SNR (dB)"].values[0]
                        if total_snr_db>best_snr:
                            best_latency, best_snr, best_path, best_channel = total_latency, total_snr_db, path, channel
        return best_snr, best_latency, best_path, best_channel

    def find_best_latency(self, source_label, dest_label, signal_power):
        best_path = None
        best_latency = float('inf')
        best_snr = float('-inf')
        best_channel = None
        paths = self.find_paths(source_label, dest_label)
        for path in paths:
            for channel in range(10):
                if self.is_channel_free_along_path(path, channel):
                    row = self._weighted_paths[self._weighted_paths["Path"] == "->".join(path)]
                    if not row.empty:
                        total_latency = row["Total Latency (s)"].values[0]
                        total_snr_db = row["SNR (dB)"].values[0]
                        if total_latency < best_latency:
                            best_latency, best_snr, best_path, best_channel = total_latency, total_snr_db, path, channel
        return best_snr, best_latency, best_path, best_channel

    def is_channel_free_along_path(self, path, channel):
        line_labels=[path[i]+path[i+1] for i in range(len(path)-1)]
        for line_label in line_labels:
            line=self.lines[line_label]
            if 0 <= channel < len(line.state):
                f=line.get_state(channel)
                if f!=1:
                    return False
        return True

    def stream_ch_sm(self, connections, label):
        best_snrs=[]
        best_latencies=[]
        best_paths=[]
        best_bit_rates=[]

        for con in connections:
            signal_power = con.signal_power
            source_label=con.input
            dest_label=con.output
            if label=="latency":
                best_snr, best_latency, best_path, best_channel=self.find_best_latency(source_label, dest_label, signal_power)
            elif label=="snr":
                best_snr, best_latency, best_path, best_channel=self.find_best_snr(source_label, dest_label, signal_power)
            else:
                raise ValueError("Invalid label: latency or snr only accepted.\n")

            print(f"Source: {source_label}, Destination: {dest_label}")
            print(f"Best Path: {best_path}")
            print(f"Best Latency: {best_latency}")
            print(f"Best SNR: {best_snr}")

            if best_path:

                prim=best_path[0]
                transceiver_strategy=self._nodes[prim].transceiver
                source_index = best_path.index(best_path[0])
                dest_index = best_path.index(best_path[-1])
                line_labels = [source + destination for source, destination in zip(best_path, best_path[1:])]

                if len(best_path) < 3:

                    ltph = Lightpath(con.signal_power, list(best_path), best_channel)
                    bit_rate = self.calculate_bit_rate_l(transceiver_strategy, ltph)
                    print(f"Bit Rate: {bit_rate}")
                    if bit_rate > 0:
                        self.propagate_lightpath(ltph)
                        if label == "snr": con.snr = best_snr
                        if label == "latency": con.latency = best_latency
                        best_snrs.append(lin2db(ltph.signal_power/ltph.noise_power))
                        best_latencies.append(ltph.latency)
                        best_paths.append(best_path)
                        best_bit_rates.append(bit_rate)
                        con.bit_rate = bit_rate
                        #self.print_lines_occupy(line_labels, best_channel)
                        best_path_lines = ['->'.join(line[i:i + 2]) for line in line_labels for i in
                                           range(len(line) - 1)]
                        self.route_space_occupy(label, best_path_lines, best_channel)
                        con.set_channel(best_channel)
                    else:
                        print("Bit rate not right.\n")
                        best_snrs.append(0.0)
                        best_latencies.append(0.0)
                        best_paths.append(best_path)
                        best_bit_rates.append(0.0)

                elif len(best_path) > 2:

                    ltph = Lightpath(con.signal_power, list(best_path), best_channel)
                    bit_rate = self.calculate_bit_rate_l(transceiver_strategy, ltph)
                    print(f"Bit Rate: {bit_rate}")
                    if bit_rate > 0:
                        middle_path_nodes = best_path[source_index + 1:dest_index]
                        mult_result = self.route_space_update(label, middle_path_nodes, best_channel, best_path,line_labels)

                        if mult_result == 1:

                            self.propagate_lightpath(ltph)
                            best_paths.append(best_path)
                            best_bit_rates.append(bit_rate)
                            con.bit_rate = bit_rate
                            #self.print_lines_occupy(line_labels, best_channel)
                            if label == "snr": con.snr = best_snr
                            if label == "latency": con.latency = best_latency
                            con.set_channel(best_channel)
                            best_snrs.append(lin2db(ltph.signal_power/ltph.noise_power))
                            best_latencies.append(ltph.latency)

                        elif mult_result == 0:
                            for line_label in line_labels:
                                line = self.lines[line_label]
                                print(f"Line: {line_label}")
                                state_channels = line.get_states()
                                print(f"Best channel {best_channel}: {state_channels}")
                            print("Occupied: path not available.\n")
                            best_snrs.append(0.0)
                            best_latencies.append(0.0)
                            best_paths.append(best_path)
                            best_bit_rates.append(0.0)
                        else:
                            print("Wrong multiplication.\n")
                    else:
                        print("Bit rate not right.\n")
                        best_snrs.append(0.0)
                        best_latencies.append(0.0)
                        best_paths.append(best_path)
                        best_bit_rates.append(0.0)
                else:
                    print("Best path length not good.\n")
                    best_snrs.append(0.0)
                    best_latencies.append(0.0)
                    best_paths.append(best_path)
                    best_bit_rates.append(0.0)
            else:
                print("Best path not found.\n")
                best_snrs.append(0.0)
                best_latencies.append(0.0)
                best_paths.append(best_path)
                best_bit_rates.append(0.0)
        return best_latencies, best_snrs, best_paths, best_bit_rates

    def stream_ch_sm_con(self, con, label):
        best_snrs = []
        best_latencies = []
        best_paths = []
        best_bit_rates = []

        signal_power = con.signal_power
        source_label = con.input
        dest_label = con.output
        if label == "latency":
            best_snr, best_latency, best_path, best_channel = self.find_best_latency(source_label, dest_label,
                                                                                     signal_power)
        elif label == "snr":
            best_snr, best_latency, best_path, best_channel = self.find_best_snr(source_label, dest_label,
                                                                                 signal_power)
        else:
            raise ValueError("Invalid label: latency or snr only accepted.\n")

        #print(f"Source: {source_label}, Destination: {dest_label}")
        #print(f"Best Path: {best_path}")
        #print(f"Best Latency: {best_latency}")
        #print(f"Best SNR: {best_snr}")

        if best_path:

            prim = best_path[0]
            transceiver_strategy = self._nodes[prim].transceiver
            source_index = best_path.index(best_path[0])
            dest_index = best_path.index(best_path[-1])
            line_labels = [source + destination for source, destination in zip(best_path, best_path[1:])]

            if len(best_path) < 3:

                ltph = Lightpath(con.signal_power, list(best_path), best_channel)
                bit_rate = self.calculate_bit_rate_l(transceiver_strategy, ltph)
                #print(f"Bit Rate: {bit_rate}")
                if bit_rate > 0:
                    self.propagate_lightpath(ltph)
                    best_snrs.append(lin2db(ltph.signal_power / ltph.noise_power))
                    best_latencies.append(ltph.latency)
                    best_paths.append(best_path)
                    best_bit_rates.append(bit_rate)
                    con.bit_rate = bit_rate
                    con.latency = best_latency
                    con.snr = best_snr
                    # self.print_lines_occupy(line_labels, best_channel)
                    best_path_lines = ['->'.join(line[i:i + 2]) for line in line_labels for i in
                                       range(len(line) - 1)]
                    self.route_space_occupy(label, best_path_lines, best_channel)
                    con.set_channel(best_channel)
                else:
                    #print("Bit rate not right.\n")
                    best_snrs.append(0.0)
                    best_latencies.append(0.0)
                    best_paths.append(best_path)
                    best_bit_rates.append(0.0)

            elif len(best_path) > 2:

                ltph = Lightpath(con.signal_power, list(best_path), best_channel)
                bit_rate = self.calculate_bit_rate_l(transceiver_strategy, ltph)
                #print(f"Bit Rate: {bit_rate}")
                if bit_rate > 0:

                    middle_path_nodes = best_path[source_index + 1:dest_index]
                    flag_ch=False
                    ch=0
                    for ch in range(best_channel, 10):
                        mult_result = self.route_space_update(label, middle_path_nodes, ch, best_path, line_labels)
                        if mult_result == 1:
                            flag_ch = True
                            break

                    if flag_ch:

                        self.propagate_lightpath(ltph)
                        best_paths.append(best_path)
                        best_bit_rates.append(bit_rate)
                        con.bit_rate = bit_rate
                        con.latency = best_latency
                        con.snr = best_snr
                        # self.print_lines_occupy(line_labels, best_channel)
                        con.set_channel(best_channel)
                        best_snrs.append(lin2db(ltph.signal_power / ltph.noise_power))
                        best_latencies.append(ltph.latency)

                        best_path_lines = ['->'.join(line[i:i + 2]) for line in line_labels for i in range(len(line) - 1)]
                        self.route_space_occupy(label, best_path_lines, ch)

                    else:

                        for line_label in line_labels:
                            line = self.lines[line_label]
                            # print(f"Line: {line_label}")
                            state_channels = line.get_states()
                            # print(f"Best channel {best_channel}: {state_channels}")
                            # print("Occupied: path not available.\n")
                        best_snrs.append(0.0)
                        best_latencies.append(0.0)
                        best_paths.append(best_path)
                        best_bit_rates.append(0.0)

                else:
                    #print("Bit rate not right.\n")
                    best_snrs.append(0.0)
                    best_latencies.append(0.0)
                    best_paths.append(best_path)
                    best_bit_rates.append(0.0)
            else:
                #print("Best path length not good.\n")
                best_snrs.append(0.0)
                best_latencies.append(0.0)
                best_paths.append(best_path)
                best_bit_rates.append(0.0)
        else:
            #print("Best path not found.\n")
            best_snrs.append(0.0)
            best_latencies.append(0.0)
            best_paths.append(best_path)
            best_bit_rates.append(0.0)
        return best_latencies, best_snrs, best_paths, best_bit_rates

    def probe(self, signal_info):
        input_node=signal_info.path[0]
        self._nodes[input_node].probe(signal_info)
        return signal_info

    def print_lines_occupy(self, line_labels, best_channel):
            for line_label in line_labels:
                line = self.lines[line_label]
                state_channels = line.get_states()
                print(f"Line: {line_label}")
                print(f"Best channel {best_channel}: {state_channels} occupied.")
                line.state_occupied(best_channel)

    def route_space(self):
        results_route_space = []
        channels = range(10)
        for source_label in self.nodes:
            for destination_label in self.nodes:
                if source_label != destination_label:
                    paths = self.find_paths(source_label, destination_label)
                    for path in paths:
                        path_string = "->".join(path)
                        result = [path_string] + [1] * len(channels)
                        results_route_space.append(result)
        columns = ["Path"] + [f"Channel{i}" for i in channels]
        route_space_df = pd.DataFrame(results_route_space, columns=columns)
        return route_space_df

    def route_space_occupy(self, label, best_path_lines, best_channel):
        if label == 'latency':
            for path_line in best_path_lines:
                self.rs_latency.loc[self.rs_latency['Path'].str.contains(path_line), f'Channel{best_channel}'] = 0
        elif label == 'snr':
            for path_line in best_path_lines:
                self.rs_snr.loc[self.rs_snr['Path'].str.contains(path_line), f'Channel{best_channel}'] = 0

    def calculate_multiplication_result(self, states, switching_states):
        mult_result = 1
        #print(len(states))
        #print(len(switching_states))
        length_min = min(len(states), len(switching_states))
        for i in range(len(states)):
            label_l, state_of_channel = states[i]
            if i < length_min:
                middle_node, state_sw = switching_states[i]
                mult_result *= state_of_channel * state_sw
            else:
                mult_result *= state_of_channel
        #print("Multiplication result:", mult_result)
        return mult_result

    def route_space_update(self, label, middle_path_nodes, best_channel, best_path, line_labels):

        states = [(line_label, self.lines[line_label].get_state(best_channel)) for line_label in line_labels]
        #print(f"States: {states}")

        states_sw = []
        for middle_path_node in middle_path_nodes:
            switching_matrix = self._switching_matrices[middle_path_node]
            middle_index = best_path.index(middle_path_node)
            if 0 < middle_index < len(best_path) - 1:
                previous_node = best_path[middle_index - 1]
                succ_node = best_path[middle_index + 1]
                state_ch = switching_matrix[previous_node][succ_node][best_channel]
                states_sw.append((middle_path_node, state_ch))
                #print(f"Switching matrix for node {middle_path_node}: {state_ch}")

        mult_result=self.calculate_multiplication_result(states, states_sw)

        if mult_result == 1.0:

            for middle_path_node in middle_path_nodes:
                switching_matrix = self._switching_matrices[middle_path_node]
                middle_index = best_path.index(middle_path_node)
                if 0 < middle_index < len(best_path) - 1:
                    previous_node = best_path[middle_index - 1]
                    succ_node = best_path[middle_index + 1]
                    #switching_matrix[previous_node][succ_node][best_channel] = 0
                    #switching_matrixs[middle_path_node] = switching_matrix

        elif mult_result == 0.0:

            for middle_path_node in middle_path_nodes:
                middle_index = best_path.index(middle_path_node)
                if 0 < middle_index < len(best_path) - 1:
                    previous_node = best_path[middle_index - 1]
                    succ_node = best_path[middle_index + 1]
                    if self._switching_matrices[middle_path_node][previous_node][succ_node][best_channel] == 0:
                        pass
                        #print( f"Path occupied: 0 in {middle_path_node} in the line {previous_node}{succ_node} in channel {best_channel}.\n")
        return mult_result

    def calculate_bit_rate_l(self, tranceiver_strategy, lightpath):

        path=lightpath.path
        Rs=lightpath.Rs
        Bn = 12.5e9  # Noise bandwidth (Hz)
        BERt = 1e-3  # Target bit error rate

        th_100 = 2 * (erfcinv(2*BERt))**2 * Rs/Bn
        th_200 = (14 / 3) * (erfcinv(BERt*3/2))**2 * Rs/Bn
        th_400 = 10 * (erfcinv( BERt*8/3))**2 * Rs/Bn

        dataframe_obj = self._weighted_paths
        #print("Searching for path:", "->".join(path))
        #print("Available paths:", dataframe_obj['Path'].unique())

        GSNR = dataframe_obj[dataframe_obj['Path'] == "->".join(path)]['SNR (dB)'].values[0]
        GSNR = db2lin(GSNR)
        #print("GSNR:", GSNR, "\n")

        if tranceiver_strategy == 'fixed-rate': #PM-QPSK
            if GSNR >= th_100:
                return 100 * 1e9
            return 0
        elif tranceiver_strategy == 'flex-rate':
            if GSNR < th_100:
                return 0
            elif th_100 <= GSNR < th_200: #PM-QPSK
                return 100 * 1e9
            elif th_200 <= GSNR < th_400: #PM-8-QAM
                return 200 * 1e9
            elif GSNR >= th_400: #PM-16-QAM
                return 400 * 1e9
        elif tranceiver_strategy == 'shannon':
            return 2 * Rs * np.log2(1 + GSNR * Rs / Bn)
        return 0

    def create_traffic_matrix(self, M):
        num_nodes = len(self.nodes)
        traffic_matrix = np.full((num_nodes, num_nodes), 100 * M, dtype=float)
        np.fill_diagonal(traffic_matrix, 0)
        return traffic_matrix

    def choose_random_connection(self, traffic_matrix):
        non_zero_requests = np.nonzero(traffic_matrix)
        if not non_zero_requests[0].size:
            return None, None
        index = random.choice(range(non_zero_requests[0].size))
        s=list(self.nodes.keys())
        sourcei, destinationi = non_zero_requests[0][index], non_zero_requests[1][index]
        source=s[sourcei]
        destination=s[destinationi]
        return source, destination, sourcei, destinationi

    def handle_connection(self, connection, traffic_matrix, path_choice, sourcei, destinationi):
        source = connection.input
        destination = connection.output
        if (traffic_matrix[sourcei][destinationi] != np.inf and traffic_matrix[sourcei][destinationi] > 0):
            best_latencies, best_snrs, best_paths, best_bit_rates = self.stream_ch_sm_con(connection, path_choice)
            bit_rate = connection.bit_rate * 1e-9
            if bit_rate !=0:
                #print(f"Connection established from {source} to {destination} with {bit_rate} Gbps.")
                traffic_matrix[sourcei][destinationi] -= bit_rate
                if traffic_matrix[sourcei][destinationi] > 0:
                    pass
                    #print(f"Connection still available with remaining {traffic_matrix[sourcei][destinationi]} Gbps.")
                else:
                    traffic_matrix[sourcei][destinationi] = np.inf
                    #print(f"Connection non available anymore.")
            else:
                return 1
        else:
            return 2
            #print(f"Insufficient bandwidth.")
        return 0

    def check_traffic_matrix(self, traffic_matrix):
        check=0
        for i in range(traffic_matrix.shape[0]):
            for j in range(traffic_matrix.shape[1]):
                if i!=j:
                    if traffic_matrix[i][j] != np.inf and traffic_matrix[i][j] > 0:
                        check=1
                        break
        return check

    def create_connections_from_traffic_matrix(self, traffic_matrix, path_choice):
        connections = []
        iterations=0
        blocking=0
        connections_made=0
        while self.check_traffic_matrix(traffic_matrix):
            iterations += 1
            source, destination, sourcei, destinationi = self.choose_random_connection(traffic_matrix)
            if source is not None and destination is not None:
                signal_power = 0.001
                connection = Connection(source, destination, signal_power)
                v=self.handle_connection(connection, traffic_matrix, path_choice, sourcei, destinationi)
                if v==1:
                    #print("Bit rate is 0.")
                    blocking=blocking+1
                    traffic_matrix[sourcei][destinationi] = -traffic_matrix[sourcei][destinationi]
                elif v==0:
                    connections.append(connection)
                    connections_made=connections_made+1
                elif v==2:
                    blocking=blocking+1
                    #print("Connection full.")
        #traffic_matrix=np.where(traffic_matrix <0, -traffic_matrix, traffic_matrix)
        traffic_matrix[traffic_matrix < 0] *= -1
        if self.check_traffic_matrix(traffic_matrix)==0:
            #print("Traffic matrix full.")
            pass
        return connections, blocking, iterations, connections_made







