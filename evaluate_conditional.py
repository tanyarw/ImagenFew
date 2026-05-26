import os
import torch
import numpy as np
import logging
from importlib import import_module
from utils.utils_args import parse_args_uncond
from data_provider.data_provider import data_provider
from metrics.conditional_metrics import calculate_conditional_metrics

def main(args):
    # Setup
    args.finetune = True # Assume we are evaluating a finetuned model
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logging.info("Starting Conditional Evaluation Pipeline...")

    # Load Data
    dataset_loader, samplers, trainsets, metadatas = data_provider(args)
    
    # Setup model handler
    handler = import_module(args.handler).Handler(args=args, rank=args.device)
    
    # Results container
    final_report = {}

    for dataset in args.train_on_datasets:
        logging.info(f"Evaluating dataset: {dataset}")
        
        # Get test data
        testset, class_label = dataset_loader.gen_dataloader(dataset)
        
        # Generate samples
        handler.model.eval()
        with torch.no_grad():
            # Using same number of samples as the test set for fair comparison
            generated_set = handler.sample(len(testset), class_label, metadatas[dataset])
        
        real_data = testset.cpu().numpy()
        gen_data = generated_set.cpu().numpy()

        # Find "Zero" in the scaled space
        # Since Dataset_Custom uses StandardScaler, zero is (0 - mean) / std
        # For simplicity in evaluation, we look for values near the minimum if they are close to 0
        # or we let the user provide a threshold. Defaulting to a small value.
        threshold = 0.05 
        
        logging.info("Calculating Conditional Metrics (Non-Zero focus)...")
        cond_results = calculate_conditional_metrics(real_data, gen_data, args.device, threshold=threshold)
        
        final_report[dataset] = cond_results
        
        # Print results immediately
        print(f"\n--- Results for {dataset} ---")
        print(f"Wet-Window Discriminative Score: {cond_results['cond_disc_score']:.4f} (Ideally close to 0)")
        print(f"Intensity Distribution JSD:     {cond_results['intensity_jsd']:.4f} (Ideally close to 0)")
        print(f"99th Percentile Ratio (Gen/Real): {cond_results['extreme_ratio']:.4f} (Ideally 1.0)")
        print(f"Avg Event Duration (Real vs Gen): {cond_results['avg_real_duration']:.2f} vs {cond_results['avg_gen_duration']:.2f}")
        print("-" * 30)

if __name__ == '__main__':
    args = parse_args_uncond()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
