#!/usr/bin/env python
# coding=utf-8
"""
Evaluation script for Something-Something V2 frame prediction model.
This script evaluates the trained model using SSIM and PSNR metrics.
"""

import argparse
import os
import torch
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from diffusers import StableDiffusionInstructPix2PixPipeline
from dataset import SomethingSomethingDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import json


def load_model(model_path):
    """
    Load the trained model from the specified path.
    """
    print(f"Loading model from {model_path}")
    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        safety_checker=None
    )
    return pipeline


def evaluate_model(model_path, dataset_dir, num_samples=100, output_dir="./evaluation_results"):
    """
    Evaluate the model on the dataset using SSIM and PSNR metrics.
    
    Args:
        model_path: Path to the trained model
        dataset_dir: Path to the dataset directory
        num_samples: Number of samples to evaluate (for faster evaluation)
        output_dir: Directory to save evaluation results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    pipeline = load_model(model_path)
    pipeline = pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    dataset = SomethingSomethingDataset(
        dataset_dir=dataset_dir,
        resolution=256,
        max_samples=num_samples
    )
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Evaluation metrics
    ssim_scores = []
    psnr_scores = []
    
    print(f"Evaluating model on {min(len(dataset), num_samples)} samples...")
    
    results = []
    
    for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        if i >= num_samples:
            break
            
        # Extract input image and target from batch
        input_image = batch['input_image'][0]  # PIL Image
        target_image = batch['target_image'][0]  # PIL Image
        edit_prompt = batch['edit_prompt'][0]  # Text prompt
        
        # Convert PIL images to numpy arrays for metric calculation
        input_np = np.array(input_image)
        target_np = np.array(target_image)
        
        # Generate prediction using the model
        try:
            # Convert input image to the right format for the pipeline
            input_pil = input_image.convert("RGB")
            
            # Generate the predicted frame
            generated_image = pipeline(
                edit_prompt,
                image=input_pil,
                num_inference_steps=20,
                image_guidance_scale=1.5,
                guidance_scale=7,
            ).images[0]
            
            # Convert generated image to numpy array
            generated_np = np.array(generated_image)
            
            # Ensure all images have the same dimensions for metric calculation
            if generated_np.shape != target_np.shape:
                # Resize generated image to match target
                from PIL import Image
                gen_pil = Image.fromarray(generated_np)
                gen_pil = gen_pil.resize((target_np.shape[1], target_np.shape[0]))
                generated_np = np.array(gen_pil)
            
            # Calculate SSIM and PSNR
            ssim_val = ssim(target_np, generated_np, channel_axis=2, data_range=255.0)
            psnr_val = psnr(target_np, generated_np, data_range=255.0)
            
            ssim_scores.append(ssim_val)
            psnr_scores.append(psnr_val)
            
            # Save sample results
            sample_result = {
                'index': i,
                'edit_prompt': edit_prompt,
                'ssim': float(ssim_val),
                'psnr': float(psnr_val)
            }
            results.append(sample_result)
            
            # Save images for visualization (optional)
            os.makedirs(f"{output_dir}/samples", exist_ok=True)
            input_image.save(f"{output_dir}/samples/input_{i:04d}.png")
            target_image.save(f"{output_dir}/samples/target_{i:04d}.png")
            generated_image.save(f"{output_dir}/samples/generated_{i:04d}.png")
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    # Calculate average metrics
    avg_ssim = np.mean(ssim_scores) if ssim_scores else 0
    avg_psnr = np.mean(psnr_scores) if psnr_scores else 0
    
    # Prepare results summary
    summary = {
        'total_samples': len(ssim_scores),
        'avg_ssim': float(avg_ssim),
        'avg_psnr': float(avg_psnr),
        'ssim_std': float(np.std(ssim_scores)) if ssim_scores else 0,
        'psnr_std': float(np.std(psnr_scores)) if psnr_scores else 0,
        'individual_results': results
    }
    
    # Save results
    with open(f"{output_dir}/evaluation_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nEvaluation Results:")
    print(f"Total samples evaluated: {summary['total_samples']}")
    print(f"Average SSIM: {summary['avg_ssim']:.4f} ± {summary['ssim_std']:.4f}")
    print(f"Average PSNR: {summary['avg_psnr']:.4f} ± {summary['psnr_std']:.4f}")
    
    return summary


def evaluate_by_task(model_path, dataset_dir, num_samples_per_task=50, output_dir="./evaluation_results"):
    """
    Evaluate the model separately for each task category.
    
    Args:
        model_path: Path to the trained model
        dataset_dir: Path to the dataset directory
        num_samples_per_task: Number of samples per task to evaluate
        output_dir: Directory to save evaluation results
    """
    from dataset import get_task_specific_dataset
    
    task_names = ['move_object', 'drop_object', 'cover_object']
    results = {}
    
    os.makedirs(output_dir, exist_ok=True)
    
    for task_name in task_names:
        print(f"\nEvaluating task: {task_name}")
        
        # Load task-specific dataset
        task_dataset = get_task_specific_dataset(
            dataset_dir=dataset_dir,
            task_name=task_name,
            resolution=256,
            max_samples=num_samples_per_task
        )
        
        if len(task_dataset) == 0:
            print(f"No samples found for task {task_name}")
            results[task_name] = {
                'total_samples': 0,
                'avg_ssim': 0,
                'avg_psnr': 0,
                'ssim_std': 0,
                'psnr_std': 0
            }
            continue
        
        # Create data loader
        dataloader = DataLoader(task_dataset, batch_size=1, shuffle=False)
        
        # Load model
        pipeline = load_model(model_path)
        pipeline = pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Evaluation metrics for this task
        ssim_scores = []
        psnr_scores = []
        
        for i, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {task_name}", leave=False)):
            if i >= num_samples_per_task:
                break
                
            # Extract input image and target from batch
            input_image = batch['input_image'][0]  # PIL Image
            target_image = batch['target_image'][0]  # PIL Image
            edit_prompt = batch['edit_prompt'][0]  # Text prompt
            
            # Convert PIL images to numpy arrays for metric calculation
            input_np = np.array(input_image)
            target_np = np.array(target_image)
            
            # Generate prediction using the model
            try:
                # Convert input image to the right format for the pipeline
                input_pil = input_image.convert("RGB")
                
                # Generate the predicted frame
                generated_image = pipeline(
                    edit_prompt,
                    image=input_pil,
                    num_inference_steps=20,
                    image_guidance_scale=1.5,
                    guidance_scale=7,
                ).images[0]
                
                # Convert generated image to numpy array
                generated_np = np.array(generated_image)
                
                # Ensure all images have the same dimensions for metric calculation
                if generated_np.shape != target_np.shape:
                    # Resize generated image to match target
                    from PIL import Image
                    gen_pil = Image.fromarray(generated_np)
                    gen_pil = gen_pil.resize((target_np.shape[1], target_np.shape[0]))
                    generated_np = np.array(gen_pil)
                
                # Calculate SSIM and PSNR
                ssim_val = ssim(target_np, generated_np, channel_axis=2, data_range=255.0)
                psnr_val = psnr(target_np, generated_np, data_range=255.0)
                
                ssim_scores.append(ssim_val)
                psnr_scores.append(psnr_val)
                
            except Exception as e:
                print(f"Error processing sample {i} for task {task_name}: {e}")
                continue
        
        # Calculate average metrics for this task
        avg_ssim = np.mean(ssim_scores) if ssim_scores else 0
        avg_psnr = np.mean(psnr_scores) if psnr_scores else 0
        
        results[task_name] = {
            'total_samples': len(ssim_scores),
            'avg_ssim': float(avg_ssim),
            'avg_psnr': float(avg_psnr),
            'ssim_std': float(np.std(ssim_scores)) if ssim_scores else 0,
            'psnr_std': float(np.std(psnr_scores)) if psnr_scores else 0
        }
    
    # Save task-specific results
    with open(f"{output_dir}/task_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTask-Specific Evaluation Results:")
    for task_name, metrics in results.items():
        print(f"{task_name}:")
        print(f"  Total samples: {metrics['total_samples']}")
        print(f"  Average SSIM: {metrics['avg_ssim']:.4f} ± {metrics['ssim_std']:.4f}")
        print(f"  Average PSNR: {metrics['avg_psnr']:.4f} ± {metrics['psnr_std']:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Something-Something V2 frame prediction model.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model directory."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to the dataset directory."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to evaluate (default: 100)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results (default: ./evaluation_results)."
    )
    parser.add_argument(
        "--by_task",
        action="store_true",
        help="Evaluate separately for each task category."
    )
    
    args = parser.parse_args()
    
    if args.by_task:
        evaluate_by_task(
            model_path=args.model_path,
            dataset_dir=args.dataset_dir,
            num_samples_per_task=args.num_samples,
            output_dir=args.output_dir
        )
    else:
        evaluate_model(
            model_path=args.model_path,
            dataset_dir=args.dataset_dir,
            num_samples=args.num_samples,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()