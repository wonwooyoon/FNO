import subprocess

def run_pflotran():
    bash_code = """
    #!/bin/bash
    shopt -s extglob
    base_dir="/home/geofluids/research/FNO"

    for i in {0..99}; do
        infile="${base_dir}/src/pflotran_code/output_hr/pflotran_${i}.in"
        mpirun -n 36 $PFLOTRAN_DIR/src/pflotran/pflotran -input_prefix "${infile%.*}"
        output_subdir="${base_dir}/src/pflotran_run/output_hr/$(basename ${infile%.*})"
        mkdir -p "${output_subdir}"
        mv ${base_dir}/src/pflotran_code/output_hr/*.h5 "${output_subdir}" 
        mv ${base_dir}/src/pflotran_code/output_hr/*.xmf "${output_subdir}"
        mv ${base_dir}/src/pflotran_code/output_hr/*.pft "${output_subdir}"
        mv ${base_dir}/src/pflotran_code/output_hr/pflotran*.dat "${output_subdir}"
        rm -rf ${base_dir}/src/pflotran_code/output_hr/*.out
    done 
    """
    subprocess.run(bash_code, shell=True, executable="/bin/bash")

if __name__ == "__main__":

    run_pflotran()
    
