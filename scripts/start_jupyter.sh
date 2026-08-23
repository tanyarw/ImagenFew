#!/bin/bash
#SBATCH --job-name=jupyter_imagen
#SBATCH --output=jupyter_imagen_%j.out
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=64G
#SBATCH --partition=gpu

# 1. Activate environment via absolute path
source /home/t/tanyawar/thesis/ImagenFew/.venv/bin/activate

# 2. Dynamically assign a port based on Slurm Job ID to avoid collisions
PORT=$(( 8000 + SLURM_JOB_ID % 1000 ))
HOSTNAME=$(hostname -s)
TOKEN="mytoken"

echo "========================================================"
echo " JOB STARTED ON COMPUTE NODE: ${HOSTNAME}"
echo "========================================================"
echo "STEP 1: Run this SSH tunnel in a NEW terminal on your Mac:"
echo ""
echo "   ssh -L ${PORT}:${HOSTNAME}:${PORT} $USER@xlogin.comp.nus.edu.sg"
echo ""
echo "   (Note: If outside NUS VPN, use -J with SoC jump host:)"
echo "   ssh -J $USER@stfjump.comp.nus.edu.sg -L ${PORT}:${HOSTNAME}:${PORT} $USER@xlogin.comp.nus.edu.sg"
echo ""
echo "STEP 2A: Open in Web Browser:"
echo "   http://localhost:${PORT}/lab?token=${TOKEN}"
echo ""
echo "STEP 2B: Or connect via VS Code (Select Existing Jupyter Server):"
echo "   http://localhost:${PORT}/?token=${TOKEN}"
echo "========================================================"

# 3. Launch JupyterLab
jupyter lab --no-browser --ip=0.0.0.0 --port=$PORT --ServerApp.token=$TOKEN --ServerApp.allow_origin='*'
