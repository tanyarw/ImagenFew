import os
import sys
import torch
import numpy as np
import pandas as pd
import logging
from importlib import import_module
from data_provider.data_provider import data_provider
from utils.utils_args import parse_args_uncond
from utils.utils import create_model_name_and_dir

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Extract years first to avoid argparse error in parse_args_uncond
    years = 10
    if '--years' in sys.argv:
        idx = sys.argv.index('--years')
        try:
            years = float(sys.argv[idx + 1])
            # Remove it so parse_args_uncond doesn't complain
            sys.argv.pop(idx)
            sys.argv.pop(idx)
        except (IndexError, ValueError):
            logging.error("Invalid value for --years. Using default (10).")

    # Load args
    args = parse_args_uncond()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.finetune = True 
    
    # Model name and directory
    create_model_name_and_dir(args)
    
    # Setup Data
    dataset_loader, samplers, trainsets, metadatas = data_provider(args)
    args.n_classes = dataset_loader.num_datasets
    
    # Setup handler
    handler = import_module(args.handler).Handler(args=args, rank=args.device)
    handler.model.eval()
    
    # Identify dataset
    dataset_name = args.train_on_datasets[0]
    if isinstance(dataset_name, dict):
        dataset_name = dataset_name['name']
        
    metadata = metadatas[dataset_name]
    
    # Get frequency from config if available, else metadata, else default
    dataset_config = next(d for d in args.datasets if d['name'] == dataset_name)
    freq = dataset_config.get('freq', metadata.get('freq', '10min'))
    logging.info(f"Detected frequency: {freq}")
    
    # Calculate steps in one non-leap year using pandas
    dr_year = pd.date_range(start='2026-01-01', end='2027-01-01', freq=freq, inclusive='left')
    steps_per_year = len(dr_year)
        
    total_steps_needed = int(years * steps_per_year)
    num_samples = int(np.ceil(total_steps_needed / args.seq_len))
    
    logging.info(f"Generating {years} years of data ({total_steps_needed} steps, {num_samples} samples of length {args.seq_len})")
    
    _, class_label = dataset_loader.gen_dataloader(dataset_name)
    
    all_generated = []
    
    # Generate in chunks
    chunk_size = 1000 
    for i in range(0, num_samples, chunk_size):
        current_n = min(chunk_size, num_samples - i)
        logging.info(f"Generating chunk {i//chunk_size + 1}/{(num_samples-1)//chunk_size + 1}...")
        with torch.no_grad():
            samples = handler.sample(current_n, class_label, metadata)
            all_generated.append(samples.cpu().numpy())
            
    # Concatenate and flatten
    generated_data = np.concatenate(all_generated, axis=0)
    continuous_series = generated_data.reshape(-1, metadata['channels'])[:total_steps_needed]
    
    # Unscale
    from data_provider.data_provider import data_dict
    dataset_config = next(d for d in args.datasets if d['name'] == dataset_name)
    dataset_config['seq_len'] = args.seq_len
    dataset_config['datasets_dir'] = args.datasets_dir
    dataset_config['flag'] = 'train'
    actual_dataset_obj = data_dict[dataset_config['data']](**dataset_config)
    unscaled_data = actual_dataset_obj.scaler.inverse_transform(continuous_series)
    
    # Clip negative values to 0 (Rainfall floor)
    unscaled_data = np.maximum(unscaled_data, 0)
    
    # Save to CSV
    output_dir = os.path.join('results', 'generated_data')
    os.makedirs(output_dir, exist_ok=True)
    
    run_id = getattr(args, 'run_id', 'unknown_run')
    output_path = os.path.join(output_dir, f'rainfall_synthetic_{years}y_{run_id}.csv')
    
    df = pd.DataFrame(unscaled_data, columns=[metadata.get('target', 'avg_rainfall')])
    df.insert(0, 'date', pd.date_range(start="2026-01-01", periods=len(df), freq=freq))
    
    df.to_csv(output_path, index=False)
    logging.info(f"Successfully saved synthetic dataset to {output_path}")

if __name__ == '__main__':
    main()
