import gdown
import zipfile
import os

# Create data directory if it doesn't exist
os.makedirs('./data', exist_ok=True)

# Google Drive ID for Datasets
dataset_id = '1EHO1EXBJYg1ohFKxmJjV548qKtbSkVhF'
gdown.download(f'https://drive.google.com/uc?id={dataset_id}', 'data.zip', quiet=False)

# Unzip and clean up
with zipfile.ZipFile('data.zip', 'r') as zip_ref:
     zip_ref.extractall('./data')
os.remove('data.zip')

# Create models directory
os.makedirs('./models_ckpt/ImagenFew/', exist_ok=True)

# Google Drive ID for Models
models_id = '16MMBjyKT6VH7YwCshp2I5RXx4FQhStiA'
gdown.download(f'https://drive.google.com/uc?id={models_id}', 'models.zip', quiet=False)

# Unzip and clean up
with zipfile.ZipFile('models.zip', 'r') as zip_ref:
     zip_ref.extractall('./models_ckpt/ImagenFew/')
os.remove('models.zip')