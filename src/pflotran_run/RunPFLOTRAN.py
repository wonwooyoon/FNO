import subprocess

def run_pflotran():
    bash_code = """
    #!/bin/bash
    shopt -s extglob
    base_dir="/home/geofluids/research/FNO"

    for i in {0..10}; do
        infile="${base_dir}/src/pflotran_code/output/pflotran_${i}.in"
        mpirun -n 36 $PFLOTRAN_DIR/src/pflotran/pflotran -input_prefix "${infile%.*}"
        output_subdir="${base_dir}/src/pflotran_run/output/$(basename ${infile%.*})"
        mkdir -p "${output_subdir}"
        mv ${base_dir}/src/pflotran_code/output/*.h5 "${output_subdir}" 
        mv ${base_dir}/src/pflotran_code/output/*.xmf "${output_subdir}"
        mv ${base_dir}/src/pflotran_code/output/*.pft "${output_subdir}"
        mv ${base_dir}/src/pflotran_code/output/pflotran*.dat "${output_subdir}"
    done 
    """
    subprocess.run(bash_code, shell=True, executable="/bin/bash")

if __name__ == "__main__":

    run_pflotran()
    