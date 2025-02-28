from pathlib import Path
import numpy as np
from core.elements import *
import random
import os
import math
from core.utils import *

ROOT_DIR=Path(__file__).parent.parent
DATA_FOLDER=ROOT_DIR/'resources'
file_input=DATA_FOLDER/'network.json'
file_input_not_full=DATA_FOLDER/'not_full_network.json'
file_input_full=DATA_FOLDER/'full_network.json'
file_input_not_fullf=DATA_FOLDER/'not_full_network_flex.json'
file_input_fullf=DATA_FOLDER/'full_network_flex.json'
file_input_not_fulls=DATA_FOLDER/'not_full_network_shannon.json'
file_input_fulls=DATA_FOLDER/'full_network_shannon.json'

if __name__ == '__main__':

    path_choices = ['snr', 'latency']
    transceiver_strategies = ['fixed-rate', 'flex-rate', 'shannon']
    num_iterations = 30

    M=list(range(1,38))

    #M = list(range(1, 38, 3))

    keys = [
        "latency_min", "latency_avg", "latency_max",
        "capacity_min", "capacity_avg", "capacity_max", "capacity_total",
        "gsnr_min", "gsnr_avg", "gsnr_max",
        "blocking_percentuals", "iters"
    ]

    medie_datas = {metrica: {strategy: {path: [] for path in path_choices} for strategy in transceiver_strategies} for metrica in keys}

    OUTPUT_FOLDER = f"output_images_congestion_mt_n"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for m in M:

        print('M=', m)

        all_latencies = {strategy: {path_choice: [[] for _ in range(num_iterations)] for path_choice in path_choices} for strategy in transceiver_strategies}
        all_snrs = {strategy: {path_choice: [[] for _ in range(num_iterations)] for path_choice in path_choices} for strategy in transceiver_strategies}
        all_bit_rates = {strategy: {path_choice: [[] for _ in range(num_iterations)] for path_choice in path_choices} for strategy in transceiver_strategies}
        capacity_total = {strategy: {path_choice: [[] for _ in range(num_iterations)] for path_choice in path_choices} for strategy in transceiver_strategies}
        blocking_percentuals={strategy: {path_choice: [[] for _ in range(num_iterations)] for path_choice in path_choices} for strategy in transceiver_strategies}
        gsnr_min = {strategy: {path_choice: [[] for _ in range(num_iterations)] for path_choice in path_choices} for strategy in transceiver_strategies}
        gsnr_max = {strategy: {path_choice: [[] for _ in range(num_iterations)] for path_choice in path_choices} for strategy in transceiver_strategies}
        gsnr_avg = {strategy: {path_choice: [[] for _ in range(num_iterations)] for path_choice in path_choices} for strategy in transceiver_strategies}
        capacity_min = {strategy: {path_choice: [[] for _ in range(num_iterations)] for path_choice in path_choices} for strategy in transceiver_strategies}
        capacity_max = {strategy: {path_choice: [[] for _ in range(num_iterations)] for path_choice in path_choices} for strategy in transceiver_strategies}
        capacity_avg = {strategy: {path_choice: [[] for _ in range(num_iterations)] for path_choice in path_choices} for strategy in transceiver_strategies}
        latency_min = {strategy: {path_choice: [[] for _ in range(num_iterations)] for path_choice in path_choices} for strategy in transceiver_strategies}
        latency_max = {strategy: {path_choice: [[] for _ in range(num_iterations)] for path_choice in path_choices} for strategy in transceiver_strategies}
        latency_avg = {strategy: {path_choice: [[] for _ in range(num_iterations)] for path_choice in path_choices} for strategy in transceiver_strategies}
        iters = {strategy: {path_choice: [[] for _ in range(num_iterations)] for path_choice in path_choices} for strategy in transceiver_strategies}
        datas = [latency_min, latency_avg, latency_max, capacity_min, capacity_avg, capacity_max, capacity_total, gsnr_min, gsnr_avg, gsnr_max, blocking_percentuals, iters]

        datas_dict = dict(zip(keys, datas))

        for path_choice in path_choices:

            print("Fixed-rate\n")

            for iteration in range(num_iterations):

                #random.seed(3)
                net=Network(file_input_full) #change the input file file_input_full or file_input_not_full
                connections=[]

                for line in net.lines.values():
                    line.channels_freed()
                tm=net.create_traffic_matrix(m)
                connections, blocking, iterations, connections_made =net.create_connections_from_traffic_matrix(tm, path_choice)
                capacity_sum=0
                for con in connections:
                    all_latencies['fixed-rate'][path_choice][iteration].append(con.latency*1000)
                    all_snrs['fixed-rate'][path_choice][iteration].append(con.snr)
                    all_bit_rates['fixed-rate'][path_choice][iteration].append(con.bit_rate*10**(-9))
                    capacity_sum += con.bit_rate
                capacity_total['fixed-rate'][path_choice][iteration].append(capacity_sum*10**(-9))
                blocking_percentual=blocking/iterations*100
                blocking_percentuals['fixed-rate'][path_choice][iteration].append(blocking_percentual)
                iters['fixed-rate'][path_choice][iteration].append(iterations)
                print(connections_made)
                gsnr_min['fixed-rate'][path_choice][iteration].append(np.min(all_snrs['fixed-rate'][path_choice][iteration]))
                gsnr_max['fixed-rate'][path_choice][iteration].append(np.max(all_snrs['fixed-rate'][path_choice][iteration]))
                gsnr_avg['fixed-rate'][path_choice][iteration].append(np.mean(all_snrs['fixed-rate'][path_choice][iteration]))
                capacity_min['fixed-rate'][path_choice][iteration].append(np.min(all_bit_rates['fixed-rate'][path_choice][iteration]))
                capacity_max['fixed-rate'][path_choice][iteration].append(np.max(all_bit_rates['fixed-rate'][path_choice][iteration]))
                capacity_avg['fixed-rate'][path_choice][iteration].append(np.mean(all_bit_rates['fixed-rate'][path_choice][iteration]))
                latency_min['fixed-rate'][path_choice][iteration].append(np.min(all_latencies['fixed-rate'][path_choice][iteration]))
                latency_max['fixed-rate'][path_choice][iteration].append(np.max(all_latencies['fixed-rate'][path_choice][iteration]))
                latency_avg['fixed-rate'][path_choice][iteration].append(np.mean(all_latencies['fixed-rate'][path_choice][iteration]))


            print("Flex-rate\n")

            for iteration in range(num_iterations):

                #random.seed(3)
                net1=Network(file_input_fullf) #change the input file file_input_fullf or file_input_not_fullf
                connections = []

                for line in net1.lines.values():
                    line.channels_freed()
                tm1 = net1.create_traffic_matrix(m)
                connections, blocking, iterations, connections_made = net1.create_connections_from_traffic_matrix(tm1, path_choice)
                capacity_sum=0
                for con in connections:
                    all_latencies['flex-rate'][path_choice][iteration].append(con.latency*1000)
                    all_snrs['flex-rate'][path_choice][iteration].append(con.snr)
                    all_bit_rates['flex-rate'][path_choice][iteration].append(con.bit_rate*10**(-9))
                    capacity_sum+=con.bit_rate
                capacity_total['flex-rate'][path_choice][iteration].append(capacity_sum*10**(-9))
                blocking_percentual=blocking/iterations*100
                blocking_percentuals['flex-rate'][path_choice][iteration].append(blocking_percentual)
                iters['flex-rate'][path_choice][iteration].append(iterations)
                print(connections_made)
                gsnr_min['flex-rate'][path_choice][iteration].append(np.min(all_snrs['flex-rate'][path_choice][iteration]))
                gsnr_max['flex-rate'][path_choice][iteration].append(np.max(all_snrs['flex-rate'][path_choice][iteration]))
                gsnr_avg['flex-rate'][path_choice][iteration].append(np.mean(all_snrs['flex-rate'][path_choice][iteration]))
                capacity_min['flex-rate'][path_choice][iteration].append(np.min(all_bit_rates['flex-rate'][path_choice][iteration]))
                capacity_max['flex-rate'][path_choice][iteration].append(np.max(all_bit_rates['flex-rate'][path_choice][iteration]))
                capacity_avg['flex-rate'][path_choice][iteration].append(np.mean(all_bit_rates['flex-rate'][path_choice][iteration]))
                latency_min['flex-rate'][path_choice][iteration].append(np.min(all_latencies['flex-rate'][path_choice][iteration]))
                latency_max['flex-rate'][path_choice][iteration].append(np.max(all_latencies['flex-rate'][path_choice][iteration]))
                latency_avg['flex-rate'][path_choice][iteration].append(np.mean(all_latencies['flex-rate'][path_choice][iteration]))

            print("Shannon\n")

            for iteration in range(num_iterations):

                #random.seed(3)
                net2=Network(file_input_fulls) #change the input file file_input_fulls or file_input_not_fulls
                connections = []

                for line in net2.lines.values():
                    line.channels_freed()
                tm2 = net2.create_traffic_matrix(m)
                connections, blocking, iterations, connections_made = net2.create_connections_from_traffic_matrix(tm2, path_choice)
                capacity_sum=0
                for con in connections:
                    all_latencies['shannon'][path_choice][iteration].append(con.latency*1000)
                    all_snrs['shannon'][path_choice][iteration].append(con.snr)
                    all_bit_rates['shannon'][path_choice][iteration].append(con.bit_rate*10**(-9))
                    capacity_sum += con.bit_rate
                capacity_total['shannon'][path_choice][iteration].append(capacity_sum*10**(-9))
                blocking_percentual=blocking/iterations*100
                blocking_percentuals['shannon'][path_choice][iteration].append(blocking_percentual)
                iters['shannon'][path_choice][iteration].append(iterations)
                print(connections_made)
                gsnr_min['shannon'][path_choice][iteration].append(np.min(all_snrs['shannon'][path_choice][iteration]))
                gsnr_max['shannon'][path_choice][iteration].append(np.max(all_snrs['shannon'][path_choice][iteration]))
                gsnr_avg['shannon'][path_choice][iteration].append(np.mean(all_snrs['shannon'][path_choice][iteration]))
                capacity_min['shannon'][path_choice][iteration].append(np.min(all_bit_rates['shannon'][path_choice][iteration]))
                capacity_max['shannon'][path_choice][iteration].append(np.max(all_bit_rates['shannon'][path_choice][iteration]))
                capacity_avg['shannon'][path_choice][iteration].append(np.mean(all_bit_rates['shannon'][path_choice][iteration]))
                latency_min['shannon'][path_choice][iteration].append(np.min(all_latencies['shannon'][path_choice][iteration]))
                latency_max['shannon'][path_choice][iteration].append(np.max(all_latencies['shannon'][path_choice][iteration]))
                latency_avg['shannon'][path_choice][iteration].append(np.mean(all_latencies['shannon'][path_choice][iteration]))

        for metrica, strategie in datas_dict.items():
            for strategy, path_choices in strategie.items():
                for path, iterazioni in path_choices.items():
                    valori = [v for iteration in iterazioni for v in iteration if isinstance(v, (int, float))]
                    medie_datas[metrica][strategy][path].append(np.mean(valori) if valori else None)

    plot_value_trends(OUTPUT_FOLDER, transceiver_strategies, 'latency', medie_datas["latency_min"], medie_datas["latency_avg"], medie_datas["latency_max"],"Latency", "ms")
    plot_value_trends(OUTPUT_FOLDER, transceiver_strategies, 'snr', medie_datas["latency_min"], medie_datas["latency_avg"], medie_datas["latency_max"], "Latency", "ms")
    #plot_value_trends_c(OUTPUT_FOLDER, transceiver_strategies, 'latency', medie_datas["capacity_min"], medie_datas["capacity_avg"], medie_datas["capacity_max"], medie_datas["capacity_total"],"Bit Rate")
    #plot_value_trends_c(OUTPUT_FOLDER, transceiver_strategies, 'snr', medie_datas["capacity_min"], medie_datas["capacity_avg"], medie_datas["capacity_max"], medie_datas["capacity_total"], "Bit Rate")
    plot_value_trends(OUTPUT_FOLDER, transceiver_strategies, 'latency', medie_datas["gsnr_min"], medie_datas["gsnr_avg"], medie_datas["gsnr_max"], "GSNR", "dB")
    plot_value_trends(OUTPUT_FOLDER, transceiver_strategies, 'snr', medie_datas["gsnr_min"], medie_datas["gsnr_avg"], medie_datas["gsnr_max"], "GSNR", "dB")

    plot_value_trends(OUTPUT_FOLDER, transceiver_strategies, 'latency', medie_datas["capacity_min"], medie_datas["capacity_avg"], medie_datas["capacity_max"],  "Bit Rate", "Gbps")
    plot_value_trends(OUTPUT_FOLDER, transceiver_strategies, 'snr', medie_datas["capacity_min"], medie_datas["capacity_avg"], medie_datas["capacity_max"], "Bit Rate", "Gbps")

    plot_value_trends_o(OUTPUT_FOLDER, transceiver_strategies, 'latency', medie_datas['capacity_total'],'Capacity total', "Gbps")
    plot_value_trends_o(OUTPUT_FOLDER, transceiver_strategies, 'snr', medie_datas['capacity_total'],'Capacity total', "Gbps")

    #plot_value_trends_o(OUTPUT_FOLDER, transceiver_strategies, 'latency', medie_datas['blocking_percentuals'], 'Blocking event', "N")
    #plot_value_trends_o(OUTPUT_FOLDER, transceiver_strategies, 'snr', medie_datas['blocking_percentuals'],'Blocking event', "N")

    #plot_value_trends_o(OUTPUT_FOLDER, transceiver_strategies, 'latency', medie_datas['iters'], 'Iterations', "N")
    #plot_value_trends_o(OUTPUT_FOLDER, transceiver_strategies, 'snr', medie_datas['iters'],'Iterations', "N")

"""
        for metrica, strategie in medie_datas.items():
            print(f"\n🔹 Metrica: {metrica}")
            for strategy, path_choices in strategie.items():
                print(f"  🔸 Strategia: {strategy}")
                for path, media in path_choices.items():
                    print(f"    ➜ Path: {path}, Media: {media}")

    print(medie_datas["latency_min"]["fixed-rate"]["latency"])
"""
