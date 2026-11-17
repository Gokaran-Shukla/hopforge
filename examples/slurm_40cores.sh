#!/bin/bash
#SBATCH --job-name=hopforge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=02:00:00
#SBATCH --output=hopforge_%j.out

# hopforge parallelises over the cores of ONE node with worker processes.
# Each worker uses one BLAS thread; the parent process too.
export HOPFORGE_NJOBS=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

module load python   # adapt to your cluster (Ibex / Shaheen)
python examples/parallel_conductance.py
