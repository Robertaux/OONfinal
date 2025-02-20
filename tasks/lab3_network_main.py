from pathlib import Path
from core.elements import *
import random

# Exercise Lab9 : non uniform switching matrix transceiver strategy

ROOT_DIR=Path(__file__).parent.parent
DATA_FOLDER=ROOT_DIR/'resources'
file_input=DATA_FOLDER/'nodes.json'
file_input_not_full=DATA_FOLDER/'nodes_not_full.json'
file_input_full=DATA_FOLDER/'nodes_full.json'
file_input_not_fullf=DATA_FOLDER/'nodes_not_full_f.json'
file_input_fullf=DATA_FOLDER/'nodes_full_f.json'
file_input_not_fulls=DATA_FOLDER/'nodes_not_full_s.json'
file_input_fulls=DATA_FOLDER/'nodes_full_s.json'

if __name__ == '__main__':

    random.seed(3)
    net=Network(file_input_not_full) #change the input file file_input_full or file_input_not_full
    signal_power_w=0.001
    connects=[]
    for _ in range(100):
        input_node=random.choice(list(net.nodes.keys()))
        output_node=random.choice(list(net.nodes.keys()))
        while input_node == output_node:
            output_node=random.choice(list(net.nodes.keys()))
        connects.append(Connection(input_node,output_node,signal_power_w))

    random.seed(3)
    net1=Network(file_input_not_fullf) #change the input file file_input_fullf or file_input_not_fullf
    signal_power_w=0.001
    connects1=[]
    for _ in range(100):
        input_node=random.choice(list(net1.nodes.keys()))
        output_node=random.choice(list(net1.nodes.keys()))
        while input_node == output_node:
            output_node=random.choice(list(net1.nodes.keys()))
        connects1.append(Connection(input_node,output_node,signal_power_w))

    random.seed(3)
    net2=Network(file_input_not_fulls) #change the input file file_input_fulls or file_input_not_fulls
    signal_power_w=0.001
    connects2=[]
    for _ in range(100):
        input_node=random.choice(list(net2.nodes.keys()))
        output_node=random.choice(list(net2.nodes.keys()))
        while input_node == output_node:
            output_node=random.choice(list(net2.nodes.keys()))
        connects2.append(Connection(input_node,output_node,signal_power_w))

    #paths_df=create_dataframe(net,signal_power_w=0.001)
    #paths_df.to_csv('weighted_path.csv', index=False)

    path_choices = ['snr', 'latency']
    transceiver_strategies=['fixed-rate', 'flex-rate', 'shannon']

    all_latencies = {strategy: {path_choice: [] for path_choice in path_choices} for strategy in transceiver_strategies}
    all_snrs = {strategy: {path_choice: [] for path_choice in path_choices} for strategy in transceiver_strategies}
    all_paths = {strategy: {path_choice: [] for path_choice in path_choices} for strategy in transceiver_strategies}
    all_bit_rates = {strategy: {path_choice: [] for path_choice in path_choices} for strategy in transceiver_strategies}
    capacity_total = {strategy: {path_choice: [] for path_choice in path_choices} for strategy in transceiver_strategies}

    print("Fixed-rate\n")
    capacity = 0
    for path_choice in path_choices:
        for line in net.lines.values():
            line.channels_freed()
        latencies, snrs, paths, bit_rates = net.stream_ch_sm(connects, path_choice)
        all_latencies["fixed-rate"][path_choice] = latencies
        all_snrs["fixed-rate"][path_choice] = snrs
        all_paths["fixed-rate"][path_choice] = paths
        all_bit_rates["fixed-rate"][path_choice] = bit_rates
        capacity = sum(bit_rates)
        capacity_total["fixed-rate"][path_choice] = capacity
        print(f"Total Capacity for fixed-rate and {path_choice}: {capacity} bps.\n")
    print(capacity_total)

    print("Flex-rate\n")
    capacity = 0
    for path_choice in path_choices:
        for line in net1.lines.values():
            line.channels_freed()
        latencies, snrs, paths, bit_rates = net1.stream_ch_sm(connects1, path_choice)
        all_latencies["flex-rate"][path_choice] = latencies
        all_snrs["flex-rate"][path_choice] = snrs
        all_paths["flex-rate"][path_choice] = paths
        all_bit_rates["flex-rate"][path_choice] = bit_rates
        capacity = sum(bit_rates)
        capacity_total["flex-rate"][path_choice] = capacity
        print(f"Total Capacity for flex-rate and {path_choice}: {capacity} bps.\n")
    print(capacity_total)

    print("Shannon\n")
    capacity = 0
    for path_choice in path_choices:
        for line in net2.lines.values():
            line.channels_freed()
        latencies, snrs, paths, bit_rates = net2.stream_ch_sm(connects2, path_choice)
        all_latencies["shannon"][path_choice] = latencies
        all_snrs["shannon"][path_choice] = snrs
        all_paths["shannon"][path_choice] = paths
        all_bit_rates["shannon"][path_choice] = bit_rates
        capacity = sum(bit_rates)
        capacity_total["shannon"][path_choice] = capacity
        print(f"Total Capacity for shannon and {path_choice}: {capacity} bps.\n")
    print(capacity_total)

    path_choice = 'snr'

    fig1, axes1 = plt.subplots(3, 3, figsize=(24, 15))

    for i, strategy in enumerate(transceiver_strategies):
        df_latencies = pd.DataFrame([0.0 if latency is None else latency for latency in all_latencies[strategy][path_choice]],columns=['Latency'])
        count_zeros_snr = all_snrs[strategy][path_choice].count(0.0)
        print("Count zeros:" ,count_zeros_snr)
        df_snrs = pd.DataFrame(all_snrs[strategy][path_choice], columns=['SNR'])
        df_bit_rates = pd.DataFrame(all_bit_rates[strategy][path_choice], columns=['Bit Rate'])

        df_latencies_z = df_latencies[df_latencies['Latency'] != 0.0]
        df_snrs_z = df_snrs[df_snrs['SNR'] != 0]
        df_bit_z_l = df_bit_rates[df_bit_rates['Bit Rate'] != 0]

        axes1[i, 0].hist(df_latencies_z['Latency'], bins=15, color='blue', alpha=1)
        #axes1[i, 0].hist(df_latencies['Latency'], bins=15, color='blue', alpha=1)
        axes1[i, 0].set_title(f'Distribution of Latencies for {strategy}')
        axes1[i, 0].set_xlabel('Latency Value')
        axes1[i, 0].set_ylabel('Frequency')
        axes1[i, 0].grid(True)

        axes1[i, 1].hist(df_snrs_z['SNR'], bins=15, color='cyan', alpha=1)
        axes1[i, 1].set_title(f'Distribution of SNRs for {strategy}')
        axes1[i, 1].set_xlabel('SNR Value')
        axes1[i, 1].set_ylabel('Frequency')
        axes1[i, 1].grid(True)

        axes1[i, 2].hist(df_bit_z_l['Bit Rate'], bins=15, color='red', alpha=1)
        axes1[i, 2].set_title(f'Distribution of Bit Rates for {strategy}')
        axes1[i, 2].set_xlabel('Bit Rate Value')
        axes1[i, 2].set_ylabel('Frequency')
        axes1[i, 2].grid(True)

        total_c=capacity_total[strategy]['snr']
        total_capacity_tbps = total_c / (1e12)
        axes1[i, 2].text(1.25, 0.5, f'Total Capacity: {total_capacity_tbps:.2f} Tbps', transform=axes1[i, 2].transAxes,verticalalignment='center', horizontalalignment='left', fontsize=12, color='black')

    fig1.suptitle(f'Distribution of Latencies, SNRs, and Bit Rates for Best SNR', fontsize=19)

    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()

    path_choice = 'latency'

    fig2, axes2 = plt.subplots(3, 3, figsize=(24, 15))

    for i, strategy in enumerate(transceiver_strategies):
        df_latencies = pd.DataFrame([0.0 if latency is None else latency for latency in all_latencies[strategy][path_choice]], columns=['Latency'])
        count_zeros_latency = all_snrs[strategy][path_choice].count(0.0)
        print("Count zeros:", count_zeros_latency)
        df_snrs = pd.DataFrame(all_snrs[strategy][path_choice], columns=['SNR'])
        df_bit_rates = pd.DataFrame(all_bit_rates[strategy][path_choice], columns=['Bit Rate'])

        df_latencies_z_l = df_latencies[df_latencies['Latency'] != 0.0]
        df_snrs_z_l = df_snrs[df_snrs['SNR'] != 0]
        df_bit_z_l = df_bit_rates[df_bit_rates['Bit Rate'] != 0]

        axes2[i, 0].hist(df_latencies_z_l['Latency'], bins=15, color='blue', alpha=1)
        #axes2[i, 0].hist(df_latencies['Latency'], bins=15, color='blue', alpha=1)
        axes2[i, 0].set_title(f'Distribution of Latencies for {strategy}')
        axes2[i, 0].set_xlabel('Latency Value')
        axes2[i, 0].set_ylabel('Frequency')
        axes2[i, 0].grid(True)

        axes2[i, 1].hist(df_snrs_z_l['SNR'], bins=15, color='cyan', alpha=1)
        axes2[i, 1].set_title(f'Distribution of SNRs for {strategy}')
        axes2[i, 1].set_xlabel('SNR Value')
        axes2[i, 1].set_ylabel('Frequency')
        axes2[i, 1].grid(True)

        axes2[i, 2].hist(df_bit_z_l['Bit Rate'], bins=15, color='red', alpha=1)
        axes2[i, 2].set_title(f'Distribution of Bit Rates for {strategy}')
        axes2[i, 2].set_xlabel('Bit Rate Value')
        axes2[i, 2].set_ylabel('Frequency')
        axes2[i, 2].grid(True)

        total_capacity_tbps = capacity_total[strategy]['latency'] / (1e12)
        axes2[i, 2].text(1.25, 0.5, f'Total Capacity: {total_capacity_tbps:.2f} Tbps', transform=axes2[i, 2].transAxes, verticalalignment='center', horizontalalignment='left', fontsize=12, color='black')

    fig2.suptitle(f'Distribution of Latencies, SNRs, and Bit Rates for Best Latency', fontsize=19)
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()

    net.rs_latency.to_csv('rs_latency.csv', index=False)
    net.rs_snr.to_csv('rs_snr.csv', index=False)






