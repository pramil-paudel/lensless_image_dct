#SBATCH -p gpu
#SBATCH --gres="gpu:titanxp:1"
#SBATCH -c 4
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH -J DCT_test_with_VGG
#SBATCH -o slurm-%j.out

python model1/main.py

