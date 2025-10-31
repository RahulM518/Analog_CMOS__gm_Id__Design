import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data from the CSV file.
try:
    df = pd.read_csv('all_0p4V.csv', header=None)
except FileNotFoundError:
    print("Error: Make sure csv file is in the same directory as the script.")
    exit()

def find_parameters(input_gm_id, input_gmro, plot=False):
    """
    Calculates the characteristic L and Id/W for a given gm/Id and gmro.
    This version finds the L-bracket based on the original data order.

    Args:
        input_gm_id (float): The input gm/Id value.
        input_gmro (float): The input gmro value.
        plot (bool): If True, generates plots to visualize the interpolation.

    Returns:
        tuple: A tuple containing the interpolated L (in nm) and the final Id/W value.
    """
    # --- 1. Data Loading and Preparation ---
    try:
        df = pd.read_csv('/home/rahulm/Desktop/RE___gm_Id____tech/all_0p2V.csv', header=None)
    except FileNotFoundError:
        print("Error: Make sure 'gmId_info_0p2V.csv' is in the same directory as the script.")
        return None, None

    # Extract data, skipping the first column label
    gm_id = df.iloc[12, 1:].dropna().to_numpy(dtype=float)
    gmro = df.iloc[14, 1:].dropna().to_numpy(dtype=float)
    id_w = df.iloc[15, 1:].dropna().to_numpy(dtype=float)
    
    # Define device parameters
    num_lines = 7
    points_per_line = 81
    lengths_str = ['180nm', '360nm', '540nm', '720nm', '900nm', '1080nm', '1260nm']
    lengths_num = np.array([float(l.replace('nm', '')) for l in lengths_str])

    # Reshape data into 2D arrays (7 lines x 81 points)
    gm_id_2d = gm_id.reshape(num_lines, points_per_line)
    gmro_2d = gmro.reshape(num_lines, points_per_line)
    id_w_2d = id_w.reshape(num_lines, points_per_line)


    # --- 2. Step 1: Get gmro for each L at the input gm/Id ---

    # For our input gm/Id, find the corresponding gmro on each L-curve
    gmro_at_input_gid = []
    for i in range(num_lines):
        x_coords = gm_id_2d[i, ::-1] # gm/Id (must be increasing)
        y_coords = gmro_2d[i, ::-1] # gmro
        interp_gmro = np.interp(input_gm_id, x_coords, y_coords)
        gmro_at_input_gid.append(interp_gmro)

    # --- 2b. NEW LOGIC: Find L by iterating through original L-brackets ---
    
    interpolated_L = None
    found_bracket = False

    # Iterate through the (L, gmro) pairs in their original order
    for i in range(num_lines - 1):
        g1 = gmro_at_input_gid[i]
        g2 = gmro_at_input_gid[i+1]
        
        l1 = lengths_num[i]
        l2 = lengths_num[i+1]
        
        # Check if the input_gmro lies between the gmro values of two *adjacent* lengths
        # This works even if g1 > g2 (non-monotonic)
        if (g1 <= input_gmro <= g2) or (g2 <= input_gmro <= g1):
            
            # We found the correct bracket (the "2 lines")
            # We can now safely interpolate using just these two points
            interpolated_L = np.interp(input_gmro, [g1, g2], [l1, l2])
            found_bracket = True
            
            print(f"\n--- Debug: Found L-bracket ---")
            print(f"  Input gmro = {input_gmro}")
            print(f"  L1 = {l1} nm  -> gmro = {g1:.2f}")
            print(f"  L2 = {l2} nm  -> gmro = {g2:.2f}")
            print(f"  Result: Interpolated L = {interpolated_L:.2f} nm")
            
            break # Stop after finding the first valid bracket

    # Handle extrapolation (if input_gmro is outside the full range)
    if not found_bracket:
        print("\n--- Warning: Could not find sequential bracket. ---")
        print(f"  Input gmro ({input_gmro}) may be outside the data range.")
        print(f"  gmro values at {input_gm_id} gm/Id: {np.round(gmro_at_input_gid, 2)}")
        print("  Falling back to full-range interpolation (may be less accurate if data is non-monotonic).")
        
        # As a fallback, use the robust sorting method from the previous script
        # This will correctly handle extrapolation.
        sort_indices = np.argsort(gmro_at_input_gid)
        sorted_gmro_at_gid = np.array(gmro_at_input_gid)[sort_indices]
        sorted_lengths_num = lengths_num[sort_indices]
        
        interpolated_L = np.interp(input_gmro, sorted_gmro_at_gid, sorted_lengths_num)


    # --- 3. Step 2: Find Id/W from L and gm/Id ---

    # For our input gm/Id, find the corresponding Id/W on each L-curve
    idw_at_input_gid = []
    for i in range(num_lines):
        x_coords = gm_id_2d[i, ::-1]
        y_coords = id_w_2d[i, ::-1] # Must also be reversed
        interp_idw = np.interp(input_gm_id, x_coords, y_coords)
        idw_at_input_gid.append(interp_idw)

    # Interpolate using the characteristic L to find the final Id/W
    # This interpolation (vs. lengths_num) is always safe as lengths_num is monotonic
    final_idw = np.interp(interpolated_L, lengths_num, idw_at_input_gid)

    # --- 4. Optional Plotting for Visualization ---
    if plot:
        # Plot for L interpolation
        plt.figure(figsize=(10, 6))
        # Plot the data in its original order
        plt.plot(lengths_num, gmro_at_input_gid, 'bo-', label=f'gmro values at gm/Id = {input_gm_id}')
        plt.plot(interpolated_L, input_gmro, 'r*', markersize=15, label=f'Your Point (L={interpolated_L:.2f} nm)')
        plt.xlabel('Transistor Length L (nm) - Original Order')
        plt.ylabel('gmro')
        plt.title('Step 1: Finding L (Sequential Bracket Method)')
        plt.grid(True)
        plt.legend()
        plt.show()

        # Plot for Id/W interpolation
        plt.figure(figsize=(10, 6))
        plt.plot(lengths_num, idw_at_input_gid, 'go-', label=f'Id/W values at gm/Id = {input_gm_id}')
        plt.plot(interpolated_L, final_idw, 'r*', markersize=15, label=f'Your Point (Id/W={final_idw:.4f})')
        plt.xlabel('Transistor Length L (nm)')
        plt.ylabel('Id/W')
        plt.title('Step 2: Finding Id/W via Interpolation')
        plt.grid(True)
        plt.legend()
        plt.show()


    return interpolated_L, final_idw

# =============================================================================
# --- EXAMPLE USAGE ---
#
# Provide the input values you want to analyze
input_gm_id = 10.0
input_gmro = 60

# Call the function. 
# Set plot=False (or remove it) to hide the graphs as you requested.
L_val, idw_val = find_parameters(input_gm_id, input_gmro, plot=False)

if L_val is not None:
    print("\n--- Results ---")
    print(f"For gm/Id = {input_gm_id} and gmro = {input_gmro}:")
    print(f"  > Interpolated Characteristic L = {L_val:.2f} nm")
    print(f"  > Corresponding Id/W = {idw_val:.4f}")
# =============================================================================
