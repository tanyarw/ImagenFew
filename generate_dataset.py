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
    
    # Load args
    args = parse_args_uncond()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.finetune = True # Assume we are using a finetuned model
    
    # User can pass --years as an extra argument or we can just hardcode/calculate
    # Since parse_args_uncond uses argparse, we might need to handle extra args manually 
    # or just use a default. Let's look for a '--years' flag.
    years = 10
    for i, arg in enumerate(sys.argv):
        if arg == '--years':
            years = float(sys.argv[i+1])
            break
            
    # Model name and directory
    create_model_name_and_dir(args)
    
    # Setup Data to get the scaler
    dataset_loader, samplers, trainsets, metadatas = data_provider(args)
    args.n_classes = dataset_loader.num_datasets
    
    # Setup handler
    handler = import_module(args.handler).Handler(args=args, rank=args.device)
    handler.model.eval()
    
    # Calculate samples needed
    # Use pandas to handle any frequency string (e.g., '5min', '10min', '1h')
    freq = metadata.get('freq', '10min')
    logging.info(f"Detected frequency: {freq}")
    
    # Calculate steps in one non-leap year
    dr_year = pd.date_range(start='2026-01-01', end='2027-01-01', freq=freq, inclusive='left')
    steps_per_year = len(dr_year)
        
    total_steps_needed = int(years * steps_per_year)
    num_samples = int(np.ceil(total_steps_needed / args.seq_len))
    
    logging.info(f"Generating {years} years of data ({total_steps_needed} steps, {num_samples} samples of length {args.seq_len})")
    
    _, class_label = dataset_loader.gen_dataloader(dataset_name)
    
    all_generated = []
    
    # Generate in chunks to avoid OOM or huge lists
    chunk_size = 1000 
    for i in range(0, num_samples, chunk_size):
        current_n = min(chunk_size, num_samples - i)
        logging.info(f"Generating chunk {i//chunk_size + 1}/{(num_samples-1)//chunk_size + 1}...")
        with torch.no_grad():
            samples = handler.sample(current_n, class_label, metadata)
            all_generated.append(samples.cpu().numpy())
            
    # Concatenate all
    generated_data = np.concatenate(all_generated, axis=0) # [num_samples, seq_len, channels]
    
    # Flatten to a continuous series
    continuous_series = generated_data.reshape(-1, metadata['channels'])[:total_steps_needed]
    
    # Unscale the data
    # Get the dataset object to use its scaler
    # Note: dataset_loader.gen_dataloader returns (test_data, class_label)
    # We need the actual dataset object for inverse_transform.
    # In data_provider, the trainsets dict contains the dataset objects.
    dataset_obj = trainsets[dataset_name]
    unscaled_data = dataset_obj.scaler.inverse_transform(continuous_series)
    
    # Save to CSV
    output_dir = os.path.join('results', 'generated_data')
    os.makedirs(output_dir, exist_ok=True)
    
    # Use run_id in filename if possible
    run_id = getattr(args, 'run_id', 'unknown_run')
    output_path = os.path.join(output_dir, f'rainfall_synthetic_{years}y_{run_id}.csv')
    
    df = pd.DataFrame(unscaled_data, columns=[metadata.get('target', 'rainfall')])
    # Add a date column starting from a dummy date
    start_date = "2026-01-01"
    df.insert(0, 'date', pd.date_range(start=start_date, periods=len(df), freq=metadata.get('freq', '10min')))
    
    df.to_csv(output_path, index=False)
    logging.info(f"Successfully saved synthetic dataset to {output_path}")

if __name__ == '__main__':
    main()
