Dataset：
Due to the large size of the dataset, it has been uploaded to Hugging Face using split zip archives. Please follow the steps below to download and extract the data:

🔗 Dataset Homepage:https://huggingface.co/datasets/gyzzzzzz/Incremental-2D



Installation
# Step 1: Install MMSegmentation
Follow the official MMSegmentation installation guide. We strongly recommend reading the MMSegmentation documentation for detailed usage.


# Step 2: Clone this repository
git clone https://github.com/gyzzzzzzzz/ISF-Mat.git
cd ISF-Mat


# Step 3: Install custom extensions
# Install mmseg-extension (if used)
git clone https://github.com/chenller/mmseg-extension.git
cd mmseg-extension
bash install.sh

# Install mmseg-2dmat
cd ./lib/mmseg-2dmat
python setup.py install
Model Configurations
Main project directory:FlashIterImage (or your local path)

Model configuration: batch/config/upernet_flash_internimage_b_in1k_768.py

Dataset configurations: batch/config/dataset/ (e.g., Graphene.py, MoS2.py)

Training
Full incremental training pipeline
Base training (thinlayer only):
The model is first trained to segment thinlayer regions using the batch/config/upernet_flash_internimage_b_in1k_768.py configuration.

Incremental training (monolayer addition):
The model incrementally learns to segment monolayer while replaying high-quality thinlayer instances to prevent catastrophic forgetting.
