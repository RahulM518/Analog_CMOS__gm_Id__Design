import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file.
try:
    df = pd.read_csv('all_1p8V.csv', header=None)
except FileNotFoundError:
    print("Error: Make sure 'gmId_info_0p2V.csv' is in the same directory as the script.")
    exit()

# Extract the data rows for gm/Id, gmro, and Id/W.
# We now start from the second column (index 1) to skip the text labels in the first column.
gm_id = df.iloc[12, 1:].dropna().to_numpy(dtype=float)
gmro = df.iloc[14, 1:].dropna().to_numpy(dtype=float)
id_w = df.iloc[15, 1:].dropna().to_numpy(dtype=float)
ft = df.iloc[23, 1:].dropna().to_numpy(dtype=float)

# Define the number of lines and points per line for plotting.
num_lines = 7
points_per_line = 81
# It looks like your data is for different transistor lengths, so we can define them for the plot legend.
lengths = ['180nm', '360nm', '540nm', '720nm', '900nm', '1080nm', '1260nm']

# --- Plot 1: gmro vs gm/Id ---
plt.figure(figsize=(10, 6))
for i in range(num_lines):
    start_index = i * points_per_line
    end_index = start_index + points_per_line
    plt.plot(gm_id[start_index:end_index], gmro[start_index:end_index], label=f'L = {lengths[i]}')

plt.xlabel('$g_m/I_d$')
plt.ylabel('$g_m r_o$')
plt.title('$g_m r_o$ vs $g_m/I_d$ for different lengths')
plt.grid(True)
plt.legend()
plt.xlim(left=4) # Set the x-axis to start from 4
plt.xlim(right=24)
plt.savefig('gmro_vs_gm_id.png') # This will save the plot as an image file.
plt.show()


# --- Plot 2: Id/W vs gm/Id ---
plt.figure(figsize=(10, 6))
for i in range(num_lines):
    start_index = i * points_per_line
    end_index = start_index + points_per_line
    plt.plot(gm_id[start_index:end_index], id_w[start_index:end_index], label=f'L = {lengths[i]}')

plt.xlabel('$g_m/I_d$')
plt.ylabel('$I_d/W$')
plt.title('$I_d/W$ vs $g_m/I_d$ for different lengths')
plt.grid(True)
plt.legend()
plt.xlim(left=4) # Set the x-axis to start from 4
plt.xlim(right=24)
plt.savefig('idw_vs_gm_id.png') # This will save the plot as an image file.
plt.show()


# --- Plot 3: ft vs gm/Id ---
plt.figure(figsize=(10, 6))
for i in range(num_lines):
    start_index = i * points_per_line
    end_index = start_index + points_per_line
    plt.plot(gm_id[start_index:end_index], ft[start_index:end_index], label=f'L = {lengths[i]}')

plt.xlabel('$g_m/I_d$')
plt.ylabel('ft')
plt.title('$ft$ vs $g_m/I_d$ for different lengths')
plt.grid(True)
plt.legend()
plt.xlim(left=2) # Set the x-axis to start from 4
plt.xlim(right=24)
plt.savefig('ft_vs_gm_id.png') # This will save the plot as an image file.
plt.show()

#now, same thing, but for PMOS also

gm_idp = df.iloc[32, 1:].dropna().to_numpy(dtype=float)
gmro_p = df.iloc[34, 1:].dropna().to_numpy(dtype=float)
id_wp = df.iloc[35, 1:].dropna().to_numpy(dtype=float)
ftp = df.iloc[43, 1:].dropna().to_numpy(dtype=float)

# Define the number of lines and points per line for plotting.
num_lines = 7
points_per_line = 81
# It looks like your data is for different transistor lengths, so we can define them for the plot legend.
lengths = ['180nm', '360nm', '540nm', '720nm', '900nm', '1080nm', '1260nm']

# --- Plot 1: gmro_p vs gm/Id ---
plt.figure(figsize=(10, 6))
for i in range(num_lines):
    start_index = i * points_per_line
    end_index = start_index + points_per_line
    plt.plot(gm_idp[start_index:end_index], gmro_p[start_index:end_index], label=f'L = {lengths[i]}')

plt.xlabel('$g_m/I_d$')
plt.ylabel('$g_m r_o$')
plt.title('$g_m r_o$ vs $g_m/I_d$ for different lengths')
plt.grid(True)
plt.legend()
plt.xlim(left=4) # Set the x-axis to start from 4
plt.xlim(right=24)
plt.savefig('gmro_p_vs_gm_idp.png') # This will save the plot as an image file.
plt.show()


# --- Plot 2: Id/W vs gm/Id ---
plt.figure(figsize=(10, 6))
for i in range(num_lines):
    start_index = i * points_per_line
    end_index = start_index + points_per_line
    plt.plot(gm_idp[start_index:end_index], id_wp[start_index:end_index], label=f'L = {lengths[i]}')

plt.xlabel('$g_m/I_d$')
plt.ylabel('$I_d/W$')
plt.title('$I_d/W$ vs $g_m/I_d$ for different lengths')
plt.grid(True)
plt.legend()
plt.xlim(left=4) # Set the x-axis to start from 4
plt.xlim(right=24)
plt.savefig('idw_vs_gm_idp.png') # This will save the plot as an image file.
plt.show()


# --- Plot 3: ft vs gm/Id ---
plt.figure(figsize=(10, 6))
for i in range(num_lines):
    start_index = i * points_per_line
    end_index = start_index + points_per_line
    plt.plot(gm_idp[start_index:end_index], ftp[start_index:end_index], label=f'L = {lengths[i]}')

plt.xlabel('$g_m/I_d$')
plt.ylabel('ftp')
plt.title('$ftp$ vs $g_m/I_d$ for different lengths')
plt.grid(True)
plt.legend()
plt.xlim(left=4) # Set the x-axis to start from 4
plt.xlim(right=24)
plt.savefig('ftp_vs_gm_idp.png') # This will save the plot as an image file.
plt.show()
