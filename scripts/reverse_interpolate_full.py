import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
# Set the path to your CSV file here
CSV_FILE_PATH = '/home/rahulm/Desktop/RE___gm_Id____tech/all_0p4V.csv'

# Set device type. True for PMOS, False for NMOS.
# This will select the correct rows to read from the CSV.
IS_PMOS = True

# --- Data Loading Function ---

def load_and_prepare_data(csv_path, is_pmos=True):
    """
    Loads and prepares the 2D data arrays from the CSV file.
    
    Returns:
        A tuple of (gm_id_2d, gmro_2d, id_w_2d, ft_w_2d, lengths_num)
        Returns (None, ...) if loading fails.
    """
    try:
        df = pd.read_csv(csv_path, header=None)
    except FileNotFoundError:
        print(f"Error: Could not find the file at '{csv_path}'")
        print("Please update the 'CSV_FILE_PATH' variable in the script.")
        return None, None, None, None, None
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        return None, None, None, None, None

    if is_pmos:
        # PMOS df lines (as in your script)
        gm_id_row, gmro_row, id_w_row, ft_row = 32, 34, 35, 43
        print("Loading data for PMOS device...")
    else:
        # NMOS df lines (as in your script's comments)
        gm_id_row, gmro_row, id_w_row, ft_row = 12, 14, 15, 23
        print("Loading data for NMOS device...")

    try:
        # Extract data, skipping the first column label
        gm_id = df.iloc[gm_id_row, 1:].dropna().to_numpy(dtype=float)
        gmro = df.iloc[gmro_row, 1:].dropna().to_numpy(dtype=float)
        id_w = df.iloc[id_w_row, 1:].dropna().to_numpy(dtype=float)
        ftp = df.iloc[ft_row, 1:].dropna().to_numpy(dtype=float)
        
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
        
        return gm_id_2d, gmro_2d, id_w_2d, ft_w_2d, lengths_num

    except Exception as e:
        print(f"Error processing data. Check CSV format and row/column indices.")
        print(f"Details: {e}")
        return None, None, None, None, None

# --- Original Function (Forward) ---

def find_parameters(input_gm_id, input_gmro, plot=False):
    """
    Calculates the characteristic L and Id/W for a given gm/Id and gmro.
    (This is your original function, slightly adapted to use the data loader)
    """
    # --- 1. Data Loading and Preparation ---
    gm_id_2d, gmro_2d, id_w_2d, ft_w_2d, lengths_num = load_and_prepare_data(CSV_FILE_PATH, IS_PMOS)
    if gm_id_2d is None:
        return None, None, None

    num_lines = len(lengths_num)

    # --- 2. Step 1: Get gmro for each L at the input gm/Id ---
    gmro_at_input_gid = []
    for i in range(num_lines):
        x_coords = gm_id_2d[i, ::-1] # gm/Id (must be increasing)
        y_coords = gmro_2d[i, ::-1] # gmro
        interp_gmro = np.interp(input_gm_id, x_coords, y_coords)
        gmro_at_input_gid.append(interp_gmro)

    # --- 2b. Find L by iterating through original L-brackets ---
    interpolated_L = None
    found_bracket = False
    bracket_index = -1

    for i in range(num_lines - 1):
        g1, g2 = gmro_at_input_gid[i], gmro_at_input_gid[i+1]
        l1, l2 = lengths_num[i], lengths_num[i+1]
        
        if (g1 <= input_gmro <= g2) or (g2 <= input_gmro <= g1):
            interpolated_L = np.interp(input_gmro, [g1, g2], [l1, l2])
            found_bracket = True
            bracket_index = i # Store the index of the first point in the bracket
            print(f"\n--- Debug (find_parameters): Found L-bracket ---")
            print(f"  Input gmro = {input_gmro}")
            print(f"  L1 = {l1} nm  -> gmro = {g1:.2f}")
            print(f"  L2 = {l2} nm  -> gmro = {g2:.2f}")
            print(f"  Result: Interpolated L = {interpolated_L:.4f} nm")
            break

    # --- 3. Step 2: Find Id/W and fT from L and gm/Id ---
    
    # Pre-calculate all Id/W and fT values at the input_gm_id
    # We still need this for the fallback case and for plotting
    idw_at_input_gid = []
    ftp_at_input_gid = []
    for i in range(num_lines):
        x_coords = gm_id_2d[i, ::-1] # gm/Id
        y_idw_coords = id_w_2d[i, ::-1]
        y_ft_coords = ft_w_2d[i, ::-1]
        idw_at_input_gid.append(np.interp(input_gm_id, x_coords, y_idw_coords))
        ftp_at_input_gid.append(np.interp(input_gm_id, x_coords, y_ft_coords))

    if not found_bracket:
        print("\n--- Warning (find_parameters): Could not find sequential bracket. ---")
        print(f"  Input gmro ({input_gmro}) may be outside the data range.")
        print(f"  gmro values at {input_gm_id} gm/Id: {np.round(gmro_at_input_gid, 2)}")
        print("  Falling back to full-range interpolation.")
        
        # As a fallback, use the robust sorting method
        sort_indices = np.argsort(gmro_at_input_gid)
        sorted_gmro_at_gid = np.array(gmro_at_input_gid)[sort_indices]
        sorted_lengths_num = lengths_num[sort_indices]
        interpolated_L = np.interp(input_gmro, sorted_gmro_at_gid, sorted_lengths_num)

        # Use full-range interpolation for Id/W and fT as well
        final_idw = np.interp(interpolated_L, lengths_num, idw_at_input_gid)
        final_ft =  np.interp(interpolated_L, lengths_num, ftp_at_input_gid)
    
    else:
        # --- LOGIC: Use *only* the bracket for Id/W and fT interpolation ---
        
        # Get the values for L1 (at bracket_index)
        l1 = lengths_num[bracket_index]
        idw1 = idw_at_input_gid[bracket_index]
        ft1 = ftp_at_input_gid[bracket_index]
        
        # Get the values for L2 (at bracket_index + 1)
        l2 = lengths_num[bracket_index + 1]
        idw2 = idw_at_input_gid[bracket_index + 1]
        ft2 = ftp_at_input_gid[bracket_index + 1]

        # Interpolate Id/W and fT *only* using this local bracket
        final_idw = np.interp(interpolated_L, [l1, l2], [idw1, idw2])
        final_ft =  np.interp(interpolated_L, [l1, l2], [ft1, ft2])

    # --- 4. Optional Plotting ---
    if plot:
        # Plotting still uses the full lists (idw_at_input_gid, etc.)
        # to show the overall context, which is good.
        plot_forward(lengths_num, gmro_at_input_gid, idw_at_input_gid, ftp_at_input_gid,
                     input_gm_id, input_gmro, interpolated_L, final_idw, final_ft)

    return interpolated_L, final_idw, final_ft

# --- NEW Function (Reverse) ---

def find_specs_from_device(input_L, input_Id, input_W, plot=False):
    """
    Calculates gm/Id, gmro, and fT for a given L, Id, and W.
    This is the "reverse" operation.
    """
    # --- 0. Calculate Id/W ---
    if input_W <= 0:
        print("Error: Width (W) must be greater than 0.")
        return None, None, None
    
    input_id_w = input_Id / input_W
    print(f"\nCalculating for L={input_L:.4f}nm, Id={input_Id*1e6:.2f}uA, W={input_W*1e6:.2f}um  (Id/W = {input_id_w:.4e} A/m)")

    # --- 1. Data Loading and Preparation ---
    gm_id_2d, gmro_2d, id_w_2d, ft_w_2d, lengths_num = load_and_prepare_data(CSV_FILE_PATH, IS_PMOS)
    if gm_id_2d is None:
        return None, None, None

    num_lines = len(lengths_num)

    # --- 2. Find L-bracket for the input_L ---
    found_bracket = False
    bracket_index = -1
    for i in range(num_lines - 1):
        l1 = lengths_num[i]
        l2 = lengths_num[i+1]
        if l1 <= input_L <= l2:
            found_bracket = True
            bracket_index = i
            print(f"\n--- Debug (find_specs): Found L-bracket ---")
            print(f"  Input L = {input_L:.4f} nm")
            print(f"  L1 = {l1} nm")
            print(f"  L2 = {l2} nm")
            break
    
    # --- 3. Prepare data for plotting (needs full 7-point data) ---
    # This list is also used for the main calculation (in Step 4.1)
    gm_id_for_plotting = []
    
    # We need *a* gm/Id to calculate the gmro and fT curves for plotting.
    # Let's pre-calculate the 7-point gm/Id list.
    for i in range(num_lines):
        # *** KEY FIX HERE ***
        # We must use the *un-reversed* arrays for this interpolation,
        # as id_w_2d[i] (original) is the increasing X-axis.
        # id_w_2d[i, ::-1] was a *decreasing* X-axis, giving bad results.
        x_coords_idw = id_w_2d[i] # Original, increasing Id/W
        y_coords_gid = gm_id_2d[i] # Original, decreasing gm/Id (fine for Y)
        
        gm_id_for_plotting.append(np.interp(input_id_w, x_coords_idw, y_coords_gid))

    # And get a temporary 7-point gm/Id for the other plots
    # This is *only* for plotting, not calculation
    temp_final_gm_id_for_plot = np.interp(input_L, lengths_num, gm_id_for_plotting)
    gmro_for_plotting_TEMP = []
    ft_for_plotting_TEMP = []
    for i in range(num_lines):
        # This interpolation (vs gm/Id) still needs the reversed, increasing X-axis
        x_gid_increasing = gm_id_2d[i, ::-1]
        y_gmro_reversed = gmro_2d[i, ::-1]
        y_ft_reversed = ft_w_2d[i, ::-1]
        gmro_for_plotting_TEMP.append(np.interp(temp_final_gm_id_for_plot, x_gid_increasing, y_gmro_reversed))
        ft_for_plotting_TEMP.append(np.interp(temp_final_gm_id_for_plot, x_gid_increasing, y_ft_reversed))


    # --- 4. Perform Interpolation ---
    if not found_bracket:
        print("\n--- Warning (find_specs): Input L is outside the data range. ---")
        print("  Falling back to full-range (7-point) interpolation for extrapolation.")
        
        # --- "GLOBAL" 7-POINT LOGIC (GOOD FOR EXTRAPOLATION) ---
        
        # Use the gm_id_for_plotting list we already calculated
        final_gm_id = np.interp(input_L, lengths_num, gm_id_for_plotting)

        # Calculate gmro and fT curves at this final_gm_id
        gmro_at_final_gid = []
        ft_at_final_gid = []
        for i in range(num_lines):
            x_coords_gid = gm_id_2d[i, ::-1] # gm/Id (increasing)
            y_coords_gmro = gmro_2d[i, ::-1]
            y_coords_ft = ft_w_2d[i, ::-1]
            gmro_at_final_gid.append(np.interp(final_gm_id, x_coords_gid, y_coords_gmro))
            ft_at_final_gid.append(np.interp(final_gm_id, x_coords_gid, y_coords_ft))

        final_gmro = np.interp(input_L, lengths_num, gmro_at_final_gid)
        final_ft = np.interp(input_L, lengths_num, ft_at_final_gid)
        
        # Set the final plot data
        gmro_for_plotting = gmro_at_final_gid
        ft_for_plotting = ft_at_final_gid

    else:
        # --- "LOCAL" 2-POINT INTERPOLATION LOGIC (As per your request) ---
        
        # Get L1 and L2
        l1 = lengths_num[bracket_index]
        l2 = lengths_num[bracket_index + 1]

        # --- Step 4.1: Find final gm/Id ---
        # We use the pre-calculated gm_id_for_plotting list
        gm_id1 = gm_id_for_plotting[bracket_index]
        gm_id2 = gm_id_for_plotting[bracket_index + 1]
        
        # Interpolate the two gm/Id values to the input_L
        final_gm_id = np.interp(input_L, [l1, l2], [gm_id1, gm_id2])

        # --- Step 4.2: Find final gmro ---
        
        # Get gm/Id data for L1 and L2 (must be increasing X-axis)
        gm_id_line1 = gm_id_2d[bracket_index, ::-1]
        gm_id_line2 = gm_id_2d[bracket_index + 1, ::-1]
        
        # Get gmro data for L1 and L2 (reversed to match gm/Id)
        gmro_line1 = gmro_2d[bracket_index, ::-1]
        gmro_line2 = gmro_2d[bracket_index + 1, ::-1]

        # Find gmro at final_gm_id for both lines
        gmro1 = np.interp(final_gm_id, gm_id_line1, gmro_line1)
        gmro2 = np.interp(final_gm_id, gm_id_line2, gmro_line2)

        # Interpolate the two gmro values to the input_L
        final_gmro = np.interp(input_L, [l1, l2], [gmro1, gmro2])

        # --- Step 4.3: Find final fT ---
        
        # Get fT data for L1 and L2 (reversed to match gm/Id)
        ft_line1 = ft_w_2d[bracket_index, ::-1]
        ft_line2 = ft_w_2d[bracket_index + 1, ::-1]

        # Find fT at final_gm_id for both lines
        ft1 = np.interp(final_gm_id, gm_id_line1, ft_line1)
        ft2 = np.interp(final_gm_id, gm_id_line2, ft_line2)
        
        # Interpolate the two fT values to the input_L
        final_ft = np.interp(input_L, [l1, l2], [ft1, ft2])

        # --- Get data for plotting ---
        # We need to re-calculate the gmro/fT curves at the *actual* final_gm_id
        gmro_for_plotting = []
        ft_for_plotting = []
        for i in range(num_lines):
            # This interpolation (vs gm/Id) still needs the reversed, increasing X-axis
            x_gid_increasing = gm_id_2d[i, ::-1]
            y_gmro_reversed = gmro_2d[i, ::-1]
            y_ft_reversed = ft_w_2d[i, ::-1]
            gmro_for_plotting.append(np.interp(final_gm_id, x_gid_increasing, y_gmro_reversed))
            ft_for_plotting.append(np.interp(final_gm_id, x_gid_increasing, y_ft_reversed))

    # --- 5. Optional Plotting ---
    if plot:
        plot_reverse(lengths_num, gm_id_for_plotting, gmro_for_plotting, ft_for_plotting,
                     input_L, input_id_w, final_gm_id, final_gmro, final_ft)

    return final_gm_id, final_gmro, final_ft

# --- Plotting Functions (Helper) ---

def plot_forward(lengths, gmro_vals, idw_vals, ft_vals, in_gid, in_gmro, out_L, out_idw, out_ft):
    """Helper to generate plots for find_parameters"""
    plt.figure(figsize=(18, 6))

    # Plot for L interpolation
    plt.subplot(1, 3, 1)
    plt.plot(lengths, gmro_vals, 'bo-', label=f'gmro @ gm/Id = {in_gid}')
    plt.plot(out_L, in_gmro, 'r*', markersize=15, label=f'Your Point (L={out_L:.4f} nm)')
    plt.xlabel('Transistor Length L (nm)')
    plt.ylabel('gmro')
    plt.title('Step 1: Finding L')
    plt.grid(True); plt.legend()

    # Plot for Id/W interpolation
    plt.subplot(1, 3, 2)
    plt.plot(lengths, idw_vals, 'go-', label=f'Id/W @ gm/Id = {in_gid}')
    plt.plot(out_L, out_idw, 'r*', markersize=15, label=f'Your Point (Id/W={out_idw:.4e})')
    plt.xlabel('Transistor Length L (nm)')
    plt.ylabel('Id/W (A/m)')
    plt.title('Step 2: Finding Id/W')
    plt.grid(True); plt.legend()

    # Plot for fT interpolation
    plt.subplot(1, 3, 3)
    plt.plot(lengths, ft_vals, 'mo-', label=f'fT @ gm/Id = {in_gid}')
    plt.plot(out_L, out_ft, 'r*', markersize=15, label=f'Your Point (fT={out_ft:.4e} Hz)')
    plt.xlabel('Transistor Length L (nm)')
    plt.ylabel('fT (Hz)')
    plt.title('Step 3: Finding fT')
    plt.grid(True); plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_reverse(lengths, gid_vals, gmro_vals, ft_vals, in_L, in_idw, out_gid, out_gmro, out_ft):
    """Helper to generate plots for find_specs_from_device"""
    plt.figure(figsize=(18, 6))

    # Plot for gm/Id interpolation
    plt.subplot(1, 3, 1)
    plt.plot(lengths, gid_vals, 'bo-', label=f'gm/Id @ Id/W = {in_idw*1e6:.2f} uA/um')
    plt.plot(in_L, out_gid, 'r*', markersize=15, label=f'Your Point (gm/Id={out_gid:.2f})')
    plt.xlabel('Transistor Length L (nm)')
    plt.ylabel('gm/Id (S/A)')
    plt.title('Step 1: Finding gm/Id')
    plt.grid(True); plt.legend()

    # Plot for gmro interpolation
    plt.subplot(1, 3, 2)
    plt.plot(lengths, gmro_vals, 'go-', label=f'gmro @ final gm/Id = {out_gid:.2f}')
    plt.plot(in_L, out_gmro, 'r*', markersize=15, label=f'Your Point (gmro={out_gmro:.2f})')
    plt.xlabel('Transistor Length L (nm)')
    plt.ylabel('gmro')
    plt.title('Step 2: Finding gmro')
    plt.grid(True); plt.legend()

    # Plot for fT interpolation
    plt.subplot(1, 3, 3)
    plt.plot(lengths, ft_vals, 'mo-', label=f'fT @ final gm/Id = {out_gid:.2f}')
    plt.plot(in_L, out_ft, 'r*', markersize=15, label=f'Your Point (fT={out_ft:.4e} Hz)')
    plt.xlabel('Transistor Length L (nm)')
    plt.ylabel('fT (Hz)')
    plt.title('Step 3: Finding fT')
    plt.grid(True); plt.legend()
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# --- EXAMPLE USAGE ---
# =============================================================================

# --- Example 1: Your original "Forward" calculation ---
# (Find L, Id/W, fT from specs)
print("\n" + "="*50)
print("--- Example 1: 'Forward' Calculation (Find L, Id/W) ---")
print("="*50)
# Use the values from your test
input_gm_id = 10.0
input_gmro = 32.35

L_val, idw_val, ft_val = find_parameters(input_gm_id, input_gmro, plot=False)

if L_val is not None:
    print("\n--- Results (Forward) ---")
    print(f"For gm/Id = {input_gm_id} and gmro = {input_gmro}:")
    print(f"  > Interpolated Characteristic L = {L_val:.4f} nm")
    print(f"  > Corresponding Id/W = {idw_val:.4e} (A/m)")
    print(f"  > Corresponding fT = {ft_val:.4e} Hz")


# --- Example 2: New "Reverse" calculation (Round-trip test) ---
# (Find gm/Id, gmro, fT from device size)
# We will feed the *outputs* from Example 1 back into this function
# to prove they match.
print("\n" + "="*50)
print("--- Example 2: 'Reverse' Calculation (Round-trip test) ---")
print("="*50)

if L_val is not None:
    # Use the L we just found
    my_L = L_val      # e.g., 180.04 nm

    # To test the Id/W, we must use the *exact* value from the forward calc.
    # We can create this by setting a W and calculating the required Id.
    # Let's use W = 10um (10e-6 m)
    my_W = 1.13e-6       # 10 um
    # Calculate the Id that gives the *exact* Id/W ratio
    my_Id = idw_val* my_W  # e.g., 8.8348 * 10e-6 = 88.348uA

    print(f"--- Info: Using forward-calculated L = {my_L:.4f} nm ---")
    print(f"--- Info: Using forward-calculated Id/W = {idw_val:.4e} A/m ---")
    print(f"--- (Achieved with W = {my_W*1e6:.2f} um and Id = {my_Id*1e6:.2f} uA) ---")

    # Call the new function
    gm_id_val_rev, gmro_val_rev, ft_val_rev = find_specs_from_device(my_L, my_Id, my_W, plot=False)

    if gm_id_val_rev is not None:
        print("\n--- Results (Reverse) ---")
        print(f"For L = {my_L:.4f} nm and Id/W = {idw_val:.4e} A/m:")
        print(f"  > Corresponding gm/Id = {gm_id_val_rev:.2f} S/A  (Original was {input_gm_id})")
        print(f"  > Corresponding gmro = {gmro_val_rev:.2f}      (Original was {input_gmro})")
        print(f"  > Corresponding fT = {ft_val_rev:.4e} Hz (Original was {ft_val:.4e})")

        # # Add a check
        print("\n--- Round-Trip Check ---")
        # Use a slightly more generous tolerance for floating point comparisons
        gm_id_match = np.isclose(gm_id_val_rev, input_gm_id, rtol=1e-5, atol=1e-3)
        gmro_match = np.isclose(gmro_val_rev, input_gmro, rtol=1e-5, atol=1e-3)
        print(f"  gm/Id matches: {gm_id_match}")
        print(f"  gmro matches:  {gmro_match}")

else:
    print("Skipping Example 2 because Example 1 failed to produce values.")
# =============================================================================

