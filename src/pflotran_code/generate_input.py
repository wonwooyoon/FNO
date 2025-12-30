import pandas as pd

# Input file path
input_file_path = "/home/geofluids/research/FNO/src/pflotran_code/basic_input_hr.txt"

# Read the input file
with open(input_file_path, 'r') as file:
    original_lines = file.readlines()

# Patterns to search and modify

search_perm = "FILENAME ../../initial_perm/output/perm_map.h5"
search_poro = "FILENAME ../../initial_perm/output/poro_map.h5"
search_clinochlore = "FILENAME ../../initial_mineral/output/clinochlore_cell.h5"
search_calcite = "FILENAME ../../initial_mineral/output/calcite_cell.h5"
search_pyrite = "FILENAME ../../initial_mineral/output/pyrite_cell.h5"
search_mont = "Smectite_MX80 	0.46894545454545455	8.5 	m^2/g"
search_seawater = "CONSTRAINT seawater_conc"
search_pressure = "LIQUID_PRESSURE 501793.22d0 ! unit: Pa"

# Read sampled data

sample = pd.read_csv("/home/geofluids/research/FNO/src/initial_others/output_hr/others.csv").values

sampled_pressure = sample[:, 0]
sampled_degra = sample[:, 1]

sampled_seawater = pd.read_csv("/home/geofluids/research/FNO/src/initial_seawater/output_hr/mixed_components.csv", header=None).values

# Generate modified files
max_files = 100

for file_index in range(max_files):

    new_lines = original_lines.copy()

    for i, line in enumerate(original_lines):
        
        if search_mont in line:
            new_lines[i] = line.replace(search_mont, f"Smectite_MX80 	{sampled_degra[file_index]}	8.5 	m^2/g\n")
            continue
        if search_clinochlore in line:
            new_lines[i] = line.replace(search_clinochlore, f"FILENAME ../../initial_mineral/output_hr/clinochlore_cell_{file_index}.h5\n")
            continue
        if search_calcite in line:
            new_lines[i] = line.replace(search_calcite, f"FILENAME ../../initial_mineral/output_hr/calcite_cell_{file_index}.h5\n")
            continue
        if search_pyrite in line:
            new_lines[i] = line.replace(search_pyrite, f"FILENAME ../../initial_mineral/output_hr/pyrite_cell_{file_index}.h5\n")
            continue
        if search_perm in line:
            new_lines[i] = line.replace(search_perm, f"FILENAME ../../initial_perm/output_hr/perm_map_{file_index}.h5\n")
            continue
        if search_poro in line:
            new_lines[i] = line.replace(search_poro, f"FILENAME ../../initial_perm/output_hr/poro_map_{file_index}.h5\n")
            continue
        if search_pressure in line:
            new_lines[i] = line.replace(search_pressure, f"LIQUID_PRESSURE {sampled_pressure[file_index]} ! unit: Pa\n")
            continue
        if search_seawater in line:
            new_lines[i+2] = f'H+ {sampled_seawater[file_index, 0]} pH\n'
            new_lines[i+3] = f'O2(aq) {sampled_seawater[file_index, 1]} PE\n'
            new_lines[i+4] = f'Al+++ {sampled_seawater[file_index, 2]} T\n'
            new_lines[i+5] = f'CO3-- {sampled_seawater[file_index, 3]} T\n'
            new_lines[i+6] = f'Ca++ {sampled_seawater[file_index, 4]} T\n'
            new_lines[i+7] = f'Cl- {sampled_seawater[file_index, 5]} Z\n'
            new_lines[i+8] = f'Fe++ {sampled_seawater[file_index, 6]} T\n'
            new_lines[i+9] = f'H4(SiO4) {sampled_seawater[file_index, 7]} T\n'
            new_lines[i+10] = f'K+ {sampled_seawater[file_index, 8]} T\n'
            new_lines[i+11] = f'Mg++ {sampled_seawater[file_index, 9]} T\n'
            new_lines[i+12] = f'Na+ {sampled_seawater[file_index, 10]} T\n'
            new_lines[i+13] = f'SO4-- {sampled_seawater[file_index, 11]} T\n'
            new_lines[i+14] = f'UO2++ {sampled_seawater[file_index, 12]} T\n'
            continue

    # Write the modified lines to a new file
    output_file_path = f"/home/geofluids/research/FNO/src/pflotran_code/output_hr/pflotran_{file_index}.in"
    with open(output_file_path, 'w') as file:
        file.writelines(new_lines)
    print(f"Generated file: {output_file_path}")

