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
    num_iterations = 150

    M=[1, 6, 19, 37]

    for m in M:
        print(m)
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

        OUTPUT_FOLDER = f"output_images_values_m_{m}"
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

        for iteration in range(num_iterations):
            #print("Fixed-rate\n")
            #random.seed(3)
            net=Network(file_input_full) #change the input file file_input_full or file_input_not_full
            connections=[]
            for path_choice in path_choices:
                for line in net.lines.values():
                    line.channels_freed()
                tm=net.create_traffic_matrix(m)
                connections, blocking, iterations, connections_made=net.create_connections_from_traffic_matrix(tm, path_choice)
                capacity_sum=0
                for con in connections:
                    all_latencies['fixed-rate'][path_choice][iteration].append(con.latency*1000)
                    all_snrs['fixed-rate'][path_choice][iteration].append(con.snr)
                    all_bit_rates['fixed-rate'][path_choice][iteration].append(con.bit_rate*10**(-9))
                    capacity_sum += con.bit_rate
                capacity_total['fixed-rate'][path_choice][iteration]=capacity_sum*10**(-9)
                blocking_percentual=blocking/iterations*100
                blocking_percentuals['fixed-rate'][path_choice][iteration]=blocking_percentual
                iters['fixed-rate'][path_choice][iteration]=iterations
                print(connections_made)
            #print("Flex-rate\n")
            #random.seed(3)
            net1=Network(file_input_fullf) #change the input file file_input_fullf or file_input_not_fullf
            connections = []
            for path_choice in path_choices:
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
                capacity_total['flex-rate'][path_choice][iteration]=capacity_sum*10**(-9)
                blocking_percentual=blocking/iterations*100
                blocking_percentuals['flex-rate'][path_choice][iteration]=blocking_percentual
                iters['flex-rate'][path_choice][iteration] = iterations
                print(connections_made)
            #print("Shannon\n")
            #random.seed(3)
            net2=Network(file_input_fulls) #change the input file file_input_fulls or file_input_not_fulls
            connections = []
            for path_choice in path_choices:
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
                capacity_total['shannon'][path_choice][iteration]=capacity_sum*10**(-9)
                blocking_percentual=blocking/iterations*100
                blocking_percentuals['shannon'][path_choice][iteration]=blocking_percentual
                iters['shannon'][path_choice][iteration] = iterations
                print(connections_made)

            for i, strategy in enumerate(transceiver_strategies):
                """print(strategy)
                print("latency")
                print(blocking_percentuals[strategy]['latency'][iteration])
                print("snr")
                print(blocking_percentuals[strategy]['snr'][iteration])"""

                gsnr_min[strategy]['latency'][iteration].append(np.min(all_snrs[strategy]['latency'][iteration]))
                gsnr_max[strategy]['latency'][iteration].append(np.max(all_snrs[strategy]['latency'][iteration]))
                gsnr_avg[strategy]['latency'][iteration].append(np.mean(all_snrs[strategy]['latency'][iteration]))
                capacity_min[strategy]['latency'][iteration].append(
                    np.min(all_bit_rates[strategy]['latency'][iteration]))
                capacity_max[strategy]['latency'][iteration].append(
                    np.max(all_bit_rates[strategy]['latency'][iteration]))
                capacity_avg[strategy]['latency'][iteration].append(
                    np.mean(all_bit_rates[strategy]['latency'][iteration]))
                latency_min[strategy]['latency'][iteration].append(
                    np.min(all_latencies[strategy]['latency'][iteration]))
                latency_max[strategy]['latency'][iteration].append(
                    np.max(all_latencies[strategy]['latency'][iteration]))
                latency_avg[strategy]['latency'][iteration].append(
                    np.mean(all_latencies[strategy]['latency'][iteration]))

                gsnr_min[strategy]['snr'][iteration].append(np.min(all_snrs[strategy]['snr'][iteration]))
                gsnr_max[strategy]['snr'][iteration].append(np.max(all_snrs[strategy]['snr'][iteration]))
                gsnr_avg[strategy]['snr'][iteration].append(np.mean(all_snrs[strategy]['snr'][iteration]))
                capacity_min[strategy]['snr'][iteration].append(np.min(all_bit_rates[strategy]['snr'][iteration]))
                capacity_max[strategy]['snr'][iteration].append(np.max(all_bit_rates[strategy]['snr'][iteration]))
                capacity_avg[strategy]['snr'][iteration].append(np.mean(all_bit_rates[strategy]['snr'][iteration]))
                latency_min[strategy]['snr'][iteration].append(np.min(all_latencies[strategy]['snr'][iteration]))
                latency_max[strategy]['snr'][iteration].append(np.max(all_latencies[strategy]['snr'][iteration]))
                latency_avg[strategy]['snr'][iteration].append(np.mean(all_latencies[strategy]['snr'][iteration]))

        #print(f'cap tot{capacity_total['shannon']['latency']}')
        #print(f'cap avg{capacity_avg['shannon']['latency']}')
        #print(f'cap max{capacity_max['shannon']['latency']}')

        fig1, axes1 = plt.subplots(3, 3, figsize=(24, 15))
        for i, strategy in enumerate(transceiver_strategies):
            df_latencies = pd.DataFrame(
                [0.0 if latency is None else latency for latency in all_latencies[strategy]['snr'][iteration]],
                columns=['Latency'])
            df_snrs = pd.DataFrame(all_snrs[strategy]['snr'][iteration], columns=['SNR'])
            df_bit_rates = pd.DataFrame(all_bit_rates[strategy]['snr'][iteration], columns=['Bit Rate'])

            df_latencies_z = df_latencies[df_latencies['Latency'] != 0.0]
            df_snrs_z = df_snrs[df_snrs['SNR'] != 0]
            df_bit_z_l = df_bit_rates[df_bit_rates['Bit Rate'] != 0]

            axes1[i, 0].hist(df_latencies_z['Latency'], bins=15, color='blue', alpha=1)
            axes1[i, 0].set_title(f'Distribution of Latencies for {strategy}')
            axes1[i, 0].set_xlabel('Latency Value (ms)')
            axes1[i, 0].set_ylabel('Frequency')
            axes1[i, 0].grid(True)

            axes1[i, 1].hist(df_snrs_z['SNR'], bins=15, color='cyan', alpha=1)
            axes1[i, 1].set_title(f'Distribution of GSNRs for {strategy}')
            axes1[i, 1].set_xlabel('GSNR Value (dB)')
            axes1[i, 1].set_ylabel('Frequency')
            axes1[i, 1].grid(True)

            axes1[i, 2].hist(df_bit_z_l['Bit Rate'], bins=15, color='red', alpha=1)
            axes1[i, 2].set_title(f'Distribution of Bit Rates for {strategy}')
            axes1[i, 2].set_xlabel('Bit Rate Value (Gbps)')
            axes1[i, 2].set_ylabel('Frequency')
            axes1[i, 2].grid(True)

        fig1.suptitle(f'Distribution of Latencies, GSNRs, and Bit Rates for Best SNR (Iteration {iteration})',
                      fontsize=19)
        fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(OUTPUT_FOLDER, f'distribution_snr_iter_{iteration}_n.png'))
        plt.close(fig1)

        fig2, axes2 = plt.subplots(3, 3, figsize=(24, 15))
        for i, strategy in enumerate(transceiver_strategies):
            df_latencies = pd.DataFrame(
                [0.0 if latency is None else latency for latency in all_latencies[strategy]['latency'][iteration]],
                columns=['Latency'])
            df_snrs = pd.DataFrame(all_snrs[strategy]['latency'][iteration], columns=['SNR'])
            df_bit_rates = pd.DataFrame(all_bit_rates[strategy]['latency'][iteration], columns=['Bit Rate'])

            df_latencies_z_l = df_latencies[df_latencies['Latency'] != 0.0]
            df_snrs_z_l = df_snrs[df_snrs['SNR'] != 0]
            df_bit_z_l = df_bit_rates[df_bit_rates['Bit Rate'] != 0]

            axes2[i, 0].hist(df_latencies_z_l['Latency'], bins=15, color='blue', alpha=1)
            axes2[i, 0].set_title(f'Distribution of Latencies for {strategy}')
            axes2[i, 0].set_xlabel('Latency Value (ms)')
            axes2[i, 0].set_ylabel('Frequency')
            axes2[i, 0].grid(True)

            axes2[i, 1].hist(df_snrs_z_l['SNR'], bins=15, color='cyan', alpha=1)
            axes2[i, 1].set_title(f'Distribution of GSNRs for {strategy}')
            axes2[i, 1].set_xlabel('GSNR Value (dB)')
            axes2[i, 1].set_ylabel('Frequency')
            axes2[i, 1].grid(True)

            axes2[i, 2].hist(df_bit_z_l['Bit Rate'], bins=15, color='red', alpha=1)
            axes2[i, 2].set_title(f'Distribution of Bit Rates for {strategy}')
            axes2[i, 2].set_xlabel('Bit Rate Value (Gbps)')
            axes2[i, 2].set_ylabel('Frequency')
            axes2[i, 2].grid(True)

        fig2.suptitle(f'Distribution of Latencies, GSNRs, and Bit Rates for Best Latency (Iteration {iteration})', fontsize=19)
        fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(OUTPUT_FOLDER, f'distribution_latency_iter_G_{iteration}.png'))
        plt.close(fig2)

        print_subplotg(OUTPUT_FOLDER, transceiver_strategies, 'latency', gsnr_min, gsnr_avg, gsnr_max, 'SNR', 'dB')
        print_subplot(OUTPUT_FOLDER, transceiver_strategies, 'latency', latency_min, latency_avg, latency_max, 'Latency', 'ms')
        print_subplot_c(OUTPUT_FOLDER, transceiver_strategies, 'latency', capacity_min, capacity_avg, capacity_max, 'Bit Rate', capacity_total)
        print_subplotg(OUTPUT_FOLDER, transceiver_strategies, 'snr', gsnr_min, gsnr_avg, gsnr_max, 'SNR', 'dB')
        print_subplot(OUTPUT_FOLDER, transceiver_strategies, 'snr', latency_min, latency_avg, latency_max, 'Latency', 'ms')
        print_subplot_c(OUTPUT_FOLDER, transceiver_strategies, 'snr', capacity_min, capacity_avg, capacity_max, 'Bit Rate', capacity_total)

        plot_blocking_percentage(OUTPUT_FOLDER, transceiver_strategies,iters, blocking_percentuals, 'latency')
        plot_blocking_percentage(OUTPUT_FOLDER, transceiver_strategies,iters, blocking_percentuals, 'snr')






