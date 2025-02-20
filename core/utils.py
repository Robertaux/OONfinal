# Use this file to define your generic methods, e.g. for plots
from core.math_utils import lin2db
import pandas as pd
import numpy as np
from core.elements import *
import os
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

def values_computed(network, path, signal_power):
    total_latency=0.0
    total_noise_power=0.0
    snr=0
    for i in range(len(path)-1):
        current_label=path[i]
        next_label=path[i+1]
        line_label=current_label+next_label
        line=network.lines.get(line_label)
        if line:
            total_latency+=line.latency_generation()
            total_noise_power+=line.noise_generation(signal_power)
            snr=signal_power/total_noise_power
    total_snr_db=lin2db(snr)
    return total_latency, total_noise_power, total_snr_db

def create_dataframe(network, signal_power_w):
    results=[]
    for source_label in network.nodes:
        for destination_label in network.nodes:
            if source_label!=destination_label:
                paths=network.find_paths(source_label, destination_label)
                for path in paths:
                    path_string="->".join(path)
                    total_latency, total_noise_power, total_snr_db=values_computed(network, path, signal_power_w)
                    results.append([path_string, total_latency, total_noise_power, total_snr_db])
    columns=["Path"  , "      Total Latency (s)"  , "    Total Noise Power (W)  "  , "  SNR (dB)  "]
    paths_df=pd.DataFrame(results, columns=columns)
    return paths_df

def print_subplot(output_folder, transceiver_strategies, path, all_min, all_avg, all_max, label, unit):
    fig1, axes1 = plt.subplots(3, 3, figsize=(24, 15))
    for i, strategy in enumerate(transceiver_strategies):
        df_min = pd.DataFrame(all_min[strategy][path],columns=[label])
        df_avg = pd.DataFrame(all_avg[strategy][path], columns=[label])
        df_max = pd.DataFrame(all_max[strategy][path], columns=[label])

        df_min_z = df_min[df_min[label] != 0.0]
        df_avg_z = df_avg[df_avg[label] != 0.0]
        df_max_z = df_max[df_max[label] != 0.0]

        axes1[i, 0].hist(np.round(df_min_z[label], decimals=5), bins=15, color='blue', alpha=1)
        axes1[i, 0].set_title(f'Distribution of minimum values of {label} for {strategy}')
        axes1[i, 0].set_xlabel(f'{label} minimum value {unit}')
        axes1[i, 0].set_ylabel('Frequency')
        axes1[i, 0].grid(True)

        axes1[i, 1].hist(np.round(df_avg_z[label], decimals=5), bins=15, color='cyan', alpha=1)
        axes1[i, 1].set_title(f'Distribution of average values of {label} for {strategy}')
        axes1[i, 1].set_xlabel(f'{label} average value {unit}')
        axes1[i, 1].set_ylabel('Frequency')
        axes1[i, 1].grid(True)

        axes1[i, 2].hist(np.round(df_max_z[label], decimals=5), bins=15, color='red', alpha=1)
        axes1[i, 2].set_title(f'Distribution of maximum values of {label}  for {strategy}')
        axes1[i, 2].set_xlabel(f'{label} maximum value {unit}')
        axes1[i, 2].set_ylabel('Frequency')
        axes1[i, 2].grid(True)

    fig1.suptitle(f'Distribution of Values of {label} {unit} in all the simulations ', fontsize=19)
    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_folder, f'distribution_{label}_{path}.png'))
    plt.close(fig1)

def print_subplot_c(output_folder, transceiver_strategies, path, all_min, all_avg, all_max, label, all_tot):
    fig1, axes1 = plt.subplots(3, 4, figsize=(24, 15))
    for i, strategy in enumerate(transceiver_strategies):
        df_min = pd.DataFrame(all_min[strategy][path],columns=[label])
        df_avg = pd.DataFrame(all_avg[strategy][path], columns=[label])
        df_max = pd.DataFrame(all_max[strategy][path], columns=[label])
        df_tot = pd.DataFrame(all_tot[strategy][path], columns=[label])

        df_min_z = df_min[df_min[label] != 0.0]
        df_avg_z = df_avg[df_avg[label] != 0.0]
        df_max_z = df_max[df_max[label] != 0.0]
        df_tot_z = df_tot[df_tot[label] != 0.0]

        axes1[i, 0].hist(np.round(df_min_z[label],decimals=5), bins=15, color='blue', alpha=1)
        axes1[i, 0].set_title(f'Distribution of minimum values of {label} for {strategy}')
        axes1[i, 0].set_xlabel(f'{label} minimum value (Gbps)')
        axes1[i, 0].set_ylabel('Frequency')
        axes1[i, 0].grid(True)

        axes1[i, 1].hist(np.round(df_avg_z[label],decimals=5), bins=15, color='lime', alpha=1)
        axes1[i, 1].set_title(f'Distribution of average values of {label} for {strategy}')
        axes1[i, 1].set_xlabel(f'{label} average value (Gbps)')
        axes1[i, 1].set_ylabel('Frequency')
        axes1[i, 1].grid(True)

        axes1[i, 2].hist(np.round(df_max_z[label],decimals=5), bins=15, color='red', alpha=1)
        axes1[i, 2].set_title(f'Distribution of maximum values of {label}  for {strategy}')
        axes1[i, 2].set_xlabel(f'{label} maximum value (Gbps)')
        axes1[i, 2].set_ylabel('Frequency')
        axes1[i, 2].grid(True)

        range_min = df_tot_z[label].min() - 1
        range_max = df_tot_z[label].max() + 1
        axes1[i, 3].hist(np.round(df_tot_z[label],decimals=5), bins=15, range=(range_min, range_max), color='cyan', alpha=1)
        axes1[i, 3].set_title(f'Distribution of total values of {label}  for {strategy}')
        axes1[i, 3].set_xlabel(f'{label} total value (Gbps)')
        axes1[i, 3].set_ylabel('Frequency')
        axes1[i, 3].grid(True)

    fig1.suptitle(f'Distribution of Values of {label} in all the simulations ', fontsize=19)
    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_folder, f'distribution_{label}_{path}.png'))
    plt.close(fig1)

def plot_blocking_percentage(output_folder, transceiver_strategies, iters, blocking_percentuals, label):
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        'fixed-rate': 'blue',
        'flex-rate': 'cyan',
        'shannon': 'red'
    }

    for strategy in transceiver_strategies:
        blocking_values = blocking_percentuals.get(strategy, {}).get(label, [])
        iterations = iters.get(strategy, {}).get(label, [])

        if not blocking_values or not iterations or len(blocking_values) != len(iterations):
            print(f"Errore nei dati per {strategy} ({label}): verificare lunghezza delle liste!")
            continue

        sorted_data = sorted(zip(iterations, blocking_values))
        iterations, blocking_values = zip(*sorted_data)
        iterations = list(iterations)
        blocking_values = list(blocking_values)

        ax.plot(iterations, blocking_values, marker='o', linestyle='-', color=colors.get(strategy, 'black'),
                label=strategy)

    ax.set_title(f'Blocking Percentage Over Iterations ({label})', fontsize=14)
    ax.set_xlabel('Iterations', fontsize=12)
    ax.set_ylabel('Blocking Percentage (%)', fontsize=12)
    ax.legend(title="Transceiver Strategies", loc="upper right", fontsize=10, title_fontsize=12)
    ax.grid(True)

    # Creazione della cartella di output se non esiste
    os.makedirs(output_folder, exist_ok=True)

    # Salvataggio del grafico
    plt.savefig(os.path.join(output_folder, f'blocking_percentage_{label}.png'))
    #plt.show()
    plt.close(fig)

def plot_capacities_total(output_folder, transceiver_strategies, capacities_total, label):
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))

    colors = ['blue', 'cyan', 'red']

    for i, strategy in enumerate(transceiver_strategies):
        df = pd.DataFrame(capacities_total[strategy][label], columns=[label])
        df_clean = df[df[label] != 0.0].dropna()
        if df_clean.empty:
            print(f"Attenzione: nessun dato valido per la strategia {strategy} ({label})")
            continue

        axes[i].hist(df_clean[label], bins=15, color=colors[i], alpha=1, edgecolor='black')
        axes[i].set_title(f'Distribution of total capacity for {label} and for {strategy}', fontsize=14)
        axes[i].set_xlabel(f'{label} Total Capacity Value (Gbps)')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True)

    fig.suptitle(f'Distribution of total capacity for {label} Across Strategies', fontsize=18)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(os.path.join(output_folder, f'distribution_total_capacity_{label}.png'))
    #plt.show()
    plt.close(fig)

def plot_bit_rate_gsnr(output_folder, transceiver_strategies, bit_rates, gsnr, label):
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        'fixed-rate': 'blue',
        'flex-rate': 'cyan',
        'shannon': 'red'
    }

    for strategy in transceiver_strategies:
        bit_rates_used = (bit_rates.get(strategy, {}).get(label, []))
        gsnr_used = gsnr.get(strategy, {}).get(label, [])

        if not isinstance(bit_rates_used, list) or not isinstance(gsnr_used, list):
            print(f"Errore: dati non validi per {strategy} ({label})")
            continue

        bit_rates_used = [item / 1e9 for sublist in bit_rates_used for item in (sublist if isinstance(sublist, list) else [sublist])]
        gsnr_used = [item for sublist in gsnr_used for item in (sublist if isinstance(sublist, list) else [sublist])]

        if len(bit_rates_used) != len(gsnr_used):
            print(f"Errore: i dati per {strategy} non hanno la stessa lunghezza! ({len(bit_rates_used)} vs {len(gsnr_used)})")
            continue

        sorted_data = sorted(zip(gsnr_used, bit_rates_used), key=lambda x: x[0])
        gsnr_used, bit_rates_used = zip(*sorted_data) if sorted_data else ([], [])
        gsnr_used = list(gsnr_used)
        bit_rates_used = list(bit_rates_used)
        ax.plot(gsnr_used, bit_rates_used, linestyle='-', color=colors.get(strategy, 'black'), label=strategy)

        if gsnr_used and bit_rates_used:
            ax.text(gsnr_used[-1], bit_rates_used[-1], strategy, fontsize=12, color=colors[strategy],
                    verticalalignment='bottom', horizontalalignment='left')

    ax.set_title(f'Bit Rates vs GSNR for different transceivers ({label})', fontsize=14)
    ax.set_xlabel('GSNR (dB)', fontsize=12)
    ax.set_ylabel('Bit Rate (Gbps)', fontsize=12)
    ax.legend(title="Transceiver Strategies", loc="upper left", fontsize=10, title_fontsize=12)
    ax.grid(True)

    plt.savefig(os.path.join(output_folder, f'bit_rate_gsnr_{label}.png'))
    plt.show()
    plt.close(fig)

def plot_conv(output_folder, transceiver, iterations, lat_means, br_means, snr_means, label):
    plt.figure(figsize=(10, 5))
    plt.plot( lat_means, label='Latency (ms)')
    plt.plot(br_means, label='Bitrate (Mbps)')
    plt.plot( snr_means, label='SNR (dB)')
    plt.xlabel('Number of iterations')
    plt.ylabel('Mean')
    plt.title('Convergence of metrics')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_folder, f'convergence_{transceiver}_{label}.png'))
    plt.close()

def plot_conv1(output_folder, transceiver, means, label, m):
    plt.figure(figsize=(10, 5))
    plt.plot(means)
    plt.xlabel('Number of iterations')
    plt.ylabel('Mean')
    plt.title('Convergence of metrics')
    plt.ylim(min(means) - 0.2, max(means) + 0.2)
    plt.grid()
    plt.savefig(os.path.join(output_folder, f'convergence_1_{transceiver}_{label}_{m}.png'))
    plt.close()

def cumulative_mean_error(iteration, value, path_choice, strategy, value_list):
    if iteration == 0:
        value_mean = np.mean([value[strategy][path_choice][iteration]])
        value_list.append(value_mean)
        return 100000
    elif iteration > 0:
        #print(value[strategy][path_choice][:iteration])
        value_mean = np.mean([val[0] for val in value[strategy][path_choice][:iteration]])
        value_mean_act = np.mean([val[0] for val in value[strategy][path_choice][:iteration + 1]])
        value_list.append(value_mean)
        return abs(value_mean - value_mean_act) / abs(value_mean)

def plot_value_trends(output_folder, transceiver_strategies, path, value_min, value_avg, value_max, type):
    fig, axes = plt.subplots(len(transceiver_strategies), 1, figsize=(10, 15), sharex=True)

    if len(transceiver_strategies) == 1:
        axes = [axes]

    for i, strategy in enumerate(transceiver_strategies):
        min_values = np.array(value_min[strategy][path])
        avg_values = np.array(value_avg[strategy][path])
        max_values = np.array(value_max[strategy][path])
        iterations = np.arange(len(min_values))

        axes[i].plot(iterations, min_values, label=f'{strategy} - Min Value', color='blue', linestyle='dashed', marker='o')
        axes[i].plot(iterations, avg_values, label=f'{strategy} - Avg Value', color='cyan', linestyle='solid', marker='s')
        axes[i].plot(iterations, max_values, label=f'{strategy} - Max Value', color='red', linestyle='dotted', marker='^')

        axes[i].set_title(f'{type} Values Evolution for {strategy} - {path}')
        axes[i].set_ylabel('Values')
        axes[i].legend()
        axes[i].grid(True)

    if isinstance(axes, list):
        axes[-1].set_xlabel('M')

    fig.suptitle(f'{type} Evolution for {path} in all Simulations', fontsize=19)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(os.path.join(output_folder, f'Values_of_{type}_trends_{path}.png'))
    plt.show()

def plot_value_trends_o(output_folder, transceiver_strategies, path, value, type):
    colors = ['blue', 'red', 'cyan']
    plt.figure(figsize=(10, 6))

    for i, strategy in enumerate(transceiver_strategies):
        values = np.array(value[strategy][path])
        iterations = np.arange(len(values))
        plt.plot(iterations, values, label=f'{strategy}', linestyle='dashed', marker='o', color=colors[i])

    plt.title(f'{type} Values Trends for {path}')
    plt.xlabel('M')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)

    plt.suptitle(f'{type} Evolution for {path} in all Simulations', fontsize=15)
    plt.tight_layout()

    plt.savefig(os.path.join(output_folder, f'Values_of_{type}_trends_{path}.png'))
    plt.show()



