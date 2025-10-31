import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def find_reverse_parameters(input_L, input_idw, plot=False):
    """
    Reverse calculation: finds gm/Id and gmro for a given L and Id/W.
    
    Args:
        input_L (float): The input transistor length L (in nm).
        input_idw (float): The input Id/W value.
        plot (bool): If True, generates plots to visualize the interpolation.
    
    Returns:
        tuple: A tuple containing the interpolated gm/Id and gmro values.
    """
    # --- 1. Data Loading and Preparation ---
    try:
        df = pd.read_csv('all_0p4V.csv', header=None)
    except FileNotFoundError:
        print("Error: Make sure the csv is in the correct path.")
        return None, None

    # PMOS data extraction (change to NMOS lines if needed)
    gm_id = df.iloc[32, 1:].dropna().to_numpy(dtype=float)
    gmro = df.iloc[34, 1:].dropna().to_numpy(dtype=float)
    id_w = df.iloc[35, 1:].dropna().to_numpy(dtype=float)
    ftp = df.iloc[43, 1:].dropna().to_numpy(dtype=float)
    
    # Define device parameters
    num_lines = 7
    points_per_line = 81
    lengths_str = ['180nm', '360nm', '540nm', '720nm', '900nm', '1080nm', '1260nm']
    lengths_num = np.array([float(l.replace('nm', '')) for l in lengths_str])

    # Reshape data into 2D arrays (7 lines x 81 points)
    gm_id_2d = gm_id.reshape(num_lines, points_per_line)
    gmro_2d = gmro.reshape(num_lines, points_per_line)
    id_w_2d = id_w.reshape(num_lines, points_per_line)
    ft_w_2d = ftp.reshape(num_lines, points_per_line)

    # --- 2. Step 1: Find the L-bracket ---
    
    # Find which two adjacent lengths bracket the input_L
    bracket_idx = None
    for i in range(num_lines - 1):
        if lengths_num[i] <= input_L <= lengths_num[i+1]:
            bracket_idx = i
            break
    
    if bracket_idx is None:
        print(f"\n--- Warning: Input L ({input_L} nm) is outside the data range ---")
        print(f"  Available L range: {lengths_num[0]} nm to {lengths_num[-1]} nm")
        return None, None
    
    l1 = lengths_num[bracket_idx]
    l2 = lengths_num[bracket_idx + 1]
    
    print(f"\n--- Debug: Found L-bracket ---")
    print(f"  Input L = {input_L} nm")
    print(f"  L1 = {l1} nm, L2 = {l2} nm")
    
    # --- 3. Step 2: For each L, find gm/Id at the input Id/W ---
    
    gm_id_at_input_idw = []
    for i in [bracket_idx, bracket_idx + 1]:
        # For this L-curve, interpolate to find gm/Id at input_idw
        # Note: id_w values typically decrease as gm/Id increases (need to check data direction)
        x_coords = id_w_2d[i, :]  # Id/W values
        y_coords = gm_id_2d[i, :]  # corresponding gm/Id values
        
        # Check if we need to reverse for monotonicity
        if x_coords[0] > x_coords[-1]:
            # Id/W is decreasing, reverse both arrays
            x_coords = x_coords[::-1]
            y_coords = y_coords[::-1]
        
        # Check if input_idw is within range for this L
        if x_coords[0] <= input_idw <= x_coords[-1]:
            interp_gm_id = np.interp(input_idw, x_coords, y_coords)
        else:
            print(f"  Warning: Id/W = {input_idw} is outside range for L = {lengths_num[i]} nm")
            print(f"    Range: [{x_coords[0]:.4f}, {x_coords[-1]:.4f}]")
            interp_gm_id = np.interp(input_idw, x_coords, y_coords)  # Allow extrapolation
        
        gm_id_at_input_idw.append(interp_gm_id)
    
    # --- 4. Step 3: Interpolate gm/Id between the two L values ---
    
    final_gm_id = np.interp(input_L, [l1, l2], gm_id_at_input_idw)
    
    print(f"\n--- Step 2-3: Finding gm/Id ---")
    print(f"  At L = {l1} nm, Id/W = {input_idw:.4f} → gm/Id = {gm_id_at_input_idw[0]:.4f}")
    print(f"  At L = {l2} nm, Id/W = {input_idw:.4f} → gm/Id = {gm_id_at_input_idw[1]:.4f}")
    print(f"  Interpolated gm/Id at L = {input_L} nm → {final_gm_id:.4f}")
    
    # --- 5. Step 4: Find gmro using the interpolated gm/Id and L ---
    
    gmro_at_L = []
    for i in [bracket_idx, bracket_idx + 1]:
        # For this L-curve, find gmro at the final_gm_id
        x_coords = gm_id_2d[i, ::-1]  # gm/Id (reversed to be increasing)
        y_coords = gmro_2d[i, ::-1]   # corresponding gmro
        
        interp_gmro = np.interp(final_gm_id, x_coords, y_coords)
        gmro_at_L.append(interp_gmro)
    
    # Interpolate gmro between the two L values
    final_gmro = np.interp(input_L, [l1, l2], gmro_at_L)
    
    print(f"\n--- Step 4: Finding gmro ---")
    print(f"  At L = {l1} nm, gm/Id = {final_gm_id:.4f} → gmro = {gmro_at_L[0]:.4f}")
    print(f"  At L = {l2} nm, gm/Id = {final_gm_id:.4f} → gmro = {gmro_at_L[1]:.4f}")
    print(f"  Interpolated gmro at L = {input_L} nm → {final_gmro:.4f}")
    
    # --- 6. Optional Plotting ---
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Id/W vs gm/Id for the two bracketing L values
        ax1 = axes[0]
        for i in [bracket_idx, bracket_idx + 1]:
            ax1.plot(id_w_2d[i, :], gm_id_2d[i, :], 'o-', 
                    label=f'L = {lengths_num[i]} nm', alpha=0.7)
        ax1.plot(input_idw, gm_id_at_input_idw[0], 'r*', markersize=15)
        ax1.plot(input_idw, gm_id_at_input_idw[1], 'r*', markersize=15)
        ax1.axhline(y=final_gm_id, color='r', linestyle='--', alpha=0.5, 
                   label=f'Final gm/Id = {final_gm_id:.2f}')
        ax1.axvline(x=input_idw, color='b', linestyle='--', alpha=0.5,
                   label=f'Input Id/W = {input_idw:.4f}')
        ax1.set_xlabel('Id/W')
        ax1.set_ylabel('gm/Id')
        ax1.set_title('Step 1: Finding gm/Id from Id/W')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: gm/Id vs gmro for the two bracketing L values
        ax2 = axes[1]
        for i in [bracket_idx, bracket_idx + 1]:
            ax2.plot(gm_id_2d[i, :], gmro_2d[i, :], 'o-', 
                    label=f'L = {lengths_num[i]} nm', alpha=0.7)
        ax2.plot(final_gm_id, gmro_at_L[0], 'r*', markersize=15)
        ax2.plot(final_gm_id, gmro_at_L[1], 'r*', markersize=15)
        ax2.axvline(x=final_gm_id, color='r', linestyle='--', alpha=0.5,
                   label=f'Final gm/Id = {final_gm_id:.2f}')
        ax2.axhline(y=final_gmro, color='g', linestyle='--', alpha=0.5,
                   label=f'Final gmro = {final_gmro:.2f}')
        ax2.set_xlabel('gm/Id')
        ax2.set_ylabel('gmro')
        ax2.set_title('Step 2: Finding gmro from gm/Id')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    return final_gm_id, final_gmro


# =============================================================================
# --- EXAMPLE USAGE ---

# Provide the input values you want to analyze
input_L = 180      # Length in nm
input_w = 1.13e-3              # w in m
input_id = 2e-3            #id in A
input_idw = input_id/input_w   # Id/W value (A/m)

# Call the reverse function
gm_id_val, gmro_val = find_reverse_parameters(input_L, input_idw, plot=True)

if gm_id_val is not None:
    print("\n" + "="*50)
    print("--- FINAL RESULTS ---")
    print(f"For L = {input_L} nm and Id/W = {input_idw}:")
    print(f"  > Interpolated gm/Id = {gm_id_val:.4f}")
    print(f"  > Interpolated gmro = {gmro_val:.4f}")
    print("="*50)
# =============================================================================
