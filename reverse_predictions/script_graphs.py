import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file.
try:
    df = pd.read_csv('/home/rahulm/Desktop/RE___gm_Id____tech/all_0p4V.csv', header=None)
except FileNotFoundError:
    print("Error: Make sure the csv is in the same directory as the script.")
    exit()

# Extract the data rows for gm/Id, gmro, and Id/W for NMOS
gm_id = df.iloc[12, 1:].dropna().to_numpy(dtype=float)
gmro = df.iloc[14, 1:].dropna().to_numpy(dtype=float)
id_w = df.iloc[15, 1:].dropna().to_numpy(dtype=float)
ft = df.iloc[23, 1:].dropna().to_numpy(dtype=float)

# Extract data for PMOS
gm_idp = df.iloc[32, 1:].dropna().to_numpy(dtype=float)
gmro_p = df.iloc[34, 1:].dropna().to_numpy(dtype=float)
id_wp = df.iloc[35, 1:].dropna().to_numpy(dtype=float)
ftp = df.iloc[43, 1:].dropna().to_numpy(dtype=float)

# Define the number of lines and points per line for plotting.
num_lines = 7
points_per_line = 81
lengths = ['180nm', '360nm', '540nm', '720nm', '900nm', '1080nm', '1260nm']

# --- NMOS Plots (3 in one figure) ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: gmro vs gm/Id
for i in range(num_lines):
    start_index = i * points_per_line
    end_index = start_index + points_per_line
    axes[0].plot(gm_id[start_index:end_index], gmro[start_index:end_index], label=f'L = {lengths[i]}')

axes[0].set_xlabel('$g_m/I_d$')
axes[0].set_ylabel('$g_m r_o$')
axes[0].set_title('$g_m r_o$ vs $g_m/I_d$ for NMOS')
axes[0].grid(True)
axes[0].legend()
axes[0].set_xlim(left=4, right=24)

# Plot 2: Id/W vs gm/Id
for i in range(num_lines):
    start_index = i * points_per_line
    end_index = start_index + points_per_line
    axes[1].plot(gm_id[start_index:end_index], id_w[start_index:end_index], label=f'L = {lengths[i]}')

axes[1].set_xlabel('$g_m/I_d$')
axes[1].set_ylabel('$I_d/W$')
axes[1].set_title('$I_d/W$ vs $g_m/I_d$ for NMOS')
axes[1].grid(True)
axes[1].legend()
axes[1].set_xlim(left=4, right=24)

# Plot 3: ft vs gm/Id
for i in range(num_lines):
    start_index = i * points_per_line
    end_index = start_index + points_per_line
    axes[2].plot(gm_id[start_index:end_index], ft[start_index:end_index], label=f'L = {lengths[i]}')

axes[2].set_xlabel('$g_m/I_d$')
axes[2].set_ylabel('ft')
axes[2].set_title('$ft$ vs $g_m/I_d$ for NMOS')
axes[2].grid(True)
axes[2].legend()
axes[2].set_xlim(left=2, right=24)

plt.tight_layout()
plt.savefig('NMOS_all_plots.png', dpi=300, bbox_inches='tight')
plt.show()


# --- PMOS Plots (3 in one figure) ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: gmro_p vs gm/Id
for i in range(num_lines):
    start_index = i * points_per_line
    end_index = start_index + points_per_line
    axes[0].plot(gm_idp[start_index:end_index], gmro_p[start_index:end_index], label=f'L = {lengths[i]}')

axes[0].set_xlabel('$g_m/I_d$')
axes[0].set_ylabel('$g_m r_o$')
axes[0].set_title('$g_m r_o$ vs $g_m/I_d$ for PMOS')
axes[0].grid(True)
axes[0].legend()
axes[0].set_xlim(left=4, right=24)

# Plot 2: Id/W vs gm/Id
for i in range(num_lines):
    start_index = i * points_per_line
    end_index = start_index + points_per_line
    axes[1].plot(gm_idp[start_index:end_index], id_wp[start_index:end_index], label=f'L = {lengths[i]}')

axes[1].set_xlabel('$g_m/I_d$')
axes[1].set_ylabel('$I_d/W$')
axes[1].set_title('$I_d/W$ vs $g_m/I_d$ for PMOS')
axes[1].grid(True)
axes[1].legend()
axes[1].set_xlim(left=4, right=24)

# Plot 3: ft vs gm/Id
for i in range(num_lines):
    start_index = i * points_per_line
    end_index = start_index + points_per_line
    axes[2].plot(gm_idp[start_index:end_index], ftp[start_index:end_index], label=f'L = {lengths[i]}')

axes[2].set_xlabel('$g_m/I_d$')
axes[2].set_ylabel('ftp')
axes[2].set_title('$ftp$ vs $g_m/I_d$ for PMOS')
axes[2].grid(True)
axes[2].legend()
axes[2].set_xlim(left=4, right=24)

plt.tight_layout()
plt.savefig('PMOS_all_plots.png', dpi=300, bbox_inches='tight')
plt.show()