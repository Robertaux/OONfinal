import numpy as np

from core.elements import *
import random
import os
import math
from core.utils import *
from pathlib import Path

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
    num_iterations = 300
    tolerance = 0.001

    M=[2, 6, 19, 37]

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

        OUTPUT_FOLDER = f"output_images_convergence_all_values_0_001_m_{m}_n"
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

        for path_choice in path_choices:

            lat_min1 = []
            lat_avg1 = []
            lat_max1 = []
            cap_min1 = []
            cap_avg1 = []
            cap_max1 = []
            cap_tot1 = []
            gr_min1 = []
            gr_avg1 = []
            gr_max1 = []
            bp1 = []
            it1 = []
            error1 = [lat_min1, lat_avg1, lat_max1, cap_min1, cap_avg1, cap_max1, cap_tot1, gr_min1, gr_avg1, gr_max1, bp1, it1]

            lat_min_f1 = []
            lat_avg_f1 = []
            lat_max_f1 = []
            cap_min_f1 = []
            cap_avg_f1 = []
            cap_max_f1 = []
            cap_tot_f1 = []
            gr_min_f1 = []
            gr_avg_f1 = []
            gr_max_f1 = []
            bp_f1 = []
            it_f1 = []
            error_f1 = [lat_min_f1, lat_avg_f1, lat_max_f1, cap_min_f1, cap_avg_f1, cap_max_f1, cap_tot_f1, gr_min_f1, gr_avg_f1, gr_max_f1, bp_f1, it_f1]

            lat_min_ls1 = []
            lat_avg_ls1 = []
            lat_max_ls1 = []
            cap_min_ls1 = []
            cap_avg_ls1 = []
            cap_max_ls1 = []
            cap_tot_ls1 = []
            gr_min_ls1 = []
            gr_avg_ls1 = []
            gr_max_ls1 = []
            bp_ls1 = []
            it_ls1 = []
            error_ls1 = [lat_min_ls1, lat_avg_ls1, lat_max_ls1, cap_min_ls1, cap_avg_ls1, cap_max_ls1, cap_tot_ls1, gr_min_ls1, gr_avg_ls1, gr_max_ls1, bp_ls1, it_ls1]

            max_index=0

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
                gsnr_min['fixed-rate'][path_choice][iteration].append(np.mean(all_snrs['fixed-rate'][path_choice][iteration]))
                gsnr_max['fixed-rate'][path_choice][iteration].append(np.max(all_snrs['fixed-rate'][path_choice][iteration]))
                gsnr_avg['fixed-rate'][path_choice][iteration].append(np.min(all_snrs['fixed-rate'][path_choice][iteration]))
                capacity_min['fixed-rate'][path_choice][iteration].append(np.min(all_bit_rates['fixed-rate'][path_choice][iteration]))
                capacity_max['fixed-rate'][path_choice][iteration].append(np.max(all_bit_rates['fixed-rate'][path_choice][iteration]))
                capacity_avg['fixed-rate'][path_choice][iteration].append(np.mean(all_bit_rates['fixed-rate'][path_choice][iteration]))
                latency_min['fixed-rate'][path_choice][iteration].append(np.min(all_latencies['fixed-rate'][path_choice][iteration]))
                latency_max['fixed-rate'][path_choice][iteration].append(np.max(all_latencies['fixed-rate'][path_choice][iteration]))
                latency_avg['fixed-rate'][path_choice][iteration].append(np.mean(all_latencies['fixed-rate'][path_choice][iteration]))

                changes = []
                for j in range(len(error1)):
                    changes.append(cumulative_mean_error(iteration, datas[j], path_choice, 'fixed-rate', error1[j]))

                if max(changes) < tolerance and iteration > 0:
                    print(f"Convergenza raggiunta per il fixed-rate {path_choice} dopo {iteration} iterazioni.\n"
                          f"{changes}")
                    max_index = changes.index(max(changes))
                    break
            plot_conv1(OUTPUT_FOLDER, 'fixed-rate', error1[max_index], path_choice, m)

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
                gsnr_min['flex-rate'][path_choice][iteration].append(np.mean(all_snrs['flex-rate'][path_choice][iteration]))
                gsnr_max['flex-rate'][path_choice][iteration].append(np.max(all_snrs['flex-rate'][path_choice][iteration]))
                gsnr_avg['flex-rate'][path_choice][iteration].append(np.min(all_snrs['flex-rate'][path_choice][iteration]))
                capacity_min['flex-rate'][path_choice][iteration].append(np.min(all_bit_rates['flex-rate'][path_choice][iteration]))
                capacity_max['flex-rate'][path_choice][iteration].append(np.max(all_bit_rates['flex-rate'][path_choice][iteration]))
                capacity_avg['flex-rate'][path_choice][iteration].append(np.mean(all_bit_rates['flex-rate'][path_choice][iteration]))
                latency_min['flex-rate'][path_choice][iteration].append(np.min(all_latencies['flex-rate'][path_choice][iteration]))
                latency_max['flex-rate'][path_choice][iteration].append(np.max(all_latencies['flex-rate'][path_choice][iteration]))
                latency_avg['flex-rate'][path_choice][iteration].append(np.mean(all_latencies['flex-rate'][path_choice][iteration]))

                changes = []
                for j in range(len(error_f1)):
                    changes.append(cumulative_mean_error(iteration, datas[j], path_choice, 'flex-rate', error_f1[j]))

                if max(changes) < tolerance and iteration > 0:
                    print(f"Convergenza raggiunta per il flex-rate {path_choice} dopo {iteration} iterazioni.\n"
                          f"{changes}")
                    max_index = changes.index(max(changes))
                    break
            plot_conv1(OUTPUT_FOLDER, 'flex-rate', error_f1[max_index], path_choice, m)

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
                gsnr_min['shannon'][path_choice][iteration].append(np.mean(all_snrs['shannon'][path_choice][iteration]))
                gsnr_max['shannon'][path_choice][iteration].append(np.max(all_snrs['shannon'][path_choice][iteration]))
                gsnr_avg['shannon'][path_choice][iteration].append(np.min(all_snrs['shannon'][path_choice][iteration]))
                capacity_min['shannon'][path_choice][iteration].append(np.min(all_bit_rates['shannon'][path_choice][iteration]))
                capacity_max['shannon'][path_choice][iteration].append(np.max(all_bit_rates['shannon'][path_choice][iteration]))
                capacity_avg['shannon'][path_choice][iteration].append(np.mean(all_bit_rates['shannon'][path_choice][iteration]))
                latency_min['shannon'][path_choice][iteration].append(np.min(all_latencies['shannon'][path_choice][iteration]))
                latency_max['shannon'][path_choice][iteration].append(np.max(all_latencies['shannon'][path_choice][iteration]))
                latency_avg['shannon'][path_choice][iteration].append(np.mean(all_latencies['shannon'][path_choice][iteration]))

                changes = []
                for j in range(len(error_ls1)):
                    changes.append(cumulative_mean_error(iteration, datas[j], path_choice, 'shannon', error_ls1[j]))

                if max(changes) < tolerance and iteration > 0:
                    print(f"Convergenza raggiunta per il shannon {path_choice} dopo {iteration} iterazioni.\n"
                          f"{changes}")
                    max_index = changes.index(max(changes))
                    break
            plot_conv1(OUTPUT_FOLDER, 'shannon', error_ls1[max_index], path_choice, m)





