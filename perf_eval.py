"""
Performance Evaluation Script for CVRP Jaya Algorithm

This script measures the dependency of solution quality and speed
relative to problem dimension and task type.

Usage:
    python performance_evaluation.py <output_path> [--vrp_dir <directory>] [--pop_size <int>] [--max_iter <int>]

Example:
    python performance_evaluation.py results.csv --vrp_dir ./instances --pop_size 30 --max_iter 200
"""

import os
import sys
import csv
import time
import glob
import logging
import argparse
from typing import List, Dict, Any, Optional

import vrplib

# Import Jaya solver from jaya.py module
from jaya import Jaya


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def extract_task_type(instance_name: str) -> str:
    """
    Extract task type from instance name.
    Type is the first letter of NAME (e.g., NAME: E-n13-k4 -> type == 'E')
    
    Args:
        instance_name: Name of the VRP instance
        
    Returns:
        First letter of the instance name (task type)
    """
    if instance_name and len(instance_name) > 0:
        return instance_name[0].upper()
    return "Unknown"


def extract_dimension(instance: Dict[str, Any]) -> int:
    """
    Extract dimension from the loaded instance.
    
    Args:
        instance: Loaded VRP instance dictionary
        
    Returns:
        Dimension of the problem (number of nodes including depot)
    """
    if 'dimension' in instance:
        return instance['dimension']
    elif 'demand' in instance:
        return len(instance['demand'])
    elif 'edge_weight' in instance:
        return len(instance['edge_weight'])
    return 0


def get_optimal_distance(instance: Dict[str, Any]) -> Optional[float]:
    """
    Extract optimal distance from instance comment if available.
    
    Args:
        instance: Loaded VRP instance dictionary
        
    Returns:
        Optimal distance or None if not available
    """
    comment = instance.get('comment', "")
    if isinstance(comment, str):
        comment = comment.strip()
        if comment.startswith('(') and comment.endswith(')'):
            comment = comment[1:-1]
        parts = comment.split(',')
        for part in parts:
            if ':' in part:
                key_value = part.strip().split(':')
                if len(key_value) == 2:
                    key = key_value[0].lower().strip()
                    if key in ('optimal value', 'best value', 'optimal', 'best'):
                        try:
                            return float(key_value[1].strip())
                        except ValueError:
                            pass
    return None


def calculate_relative_error_percent(total_distance: float, optimal_distance: Optional[float]) -> Optional[float]:
    """
    Calculate relative error percentage (|total - optimal| / optimal * 100).
    
    Args:
        total_distance: Achieved total distance
        optimal_distance: Known optimal distance
        
    Returns:
        Relative error in percent, or None if optimal is not available or zero
    """
    if optimal_distance is None or optimal_distance == 0:
        return None
    return abs(total_distance - optimal_distance) / optimal_distance * 100.0


def evaluate_instance(
    vrp_file: str,
    pop_size: int = 30,
    max_iter: int = 200
) -> Dict[str, Any]:
    """
    Evaluate a single VRP instance.
    
    Args:
        vrp_file: Path to the .vrp file
        pop_size: Population size for Jaya algorithm
        max_iter: Maximum iterations for Jaya algorithm
        
    Returns:
        Dictionary with evaluation results
    """
    file_name = os.path.basename(vrp_file)
    logger.info(f"Processing: {file_name}")
    
    # Load instance to get metadata
    instance = vrplib.read_instance(vrp_file)
    instance_name = instance.get('name', file_name)
    task_type = extract_task_type(instance_name)
    dimension = extract_dimension(instance)
    optimal_distance = get_optimal_distance(instance)
    
    logger.info(f"  Instance: {instance_name}, Type: {task_type}, Dimension: {dimension}")
    
    # Create solver and measure inference time
    solver = Jaya(pop_size=pop_size, max_iter=max_iter)
    
    start_time = time.time()
    result = solver.predict(vrp_file)
    end_time = time.time()
    
    inference_time = end_time - start_time
    total_distance = result['total_distance']
    
    # Use optimal from result if available, otherwise from instance
    if result.get('optimal_value') is not None:
        optimal_distance = float(result['optimal_value'])
    
    relative_error_percent = calculate_relative_error_percent(total_distance, optimal_distance)
    
    logger.info(f"  Total distance: {total_distance}")
    if optimal_distance:
        logger.info(f"  Optimal distance: {optimal_distance}")
        if relative_error_percent is not None:
            logger.info(f"  Relative Error (%): {relative_error_percent:.2f}%")
    logger.info(f"  Inference time: {inference_time:.2f}s")
    
    return {
        'file_name': file_name,
        'type': task_type,
        'dimension': dimension,
        'total_distance': total_distance,
        'optimal_distance': optimal_distance if optimal_distance else '',
        'relative_error_percent': round(relative_error_percent, 2) if relative_error_percent is not None else '',
        'inference_time': round(inference_time, 4)
    }


def find_vrp_files(directory: str) -> List[str]:
    """
    Find all .vrp files in the specified directory.
    
    Args:
        directory: Directory path to search
        
    Returns:
        List of paths to .vrp files
    """
    patterns = [
        os.path.join(directory, '*.vrp'),
        os.path.join(directory, '**', '*.vrp')
    ]
    
    vrp_files = []
    for pattern in patterns:
        vrp_files.extend(glob.glob(pattern, recursive=True))
    
    # Remove duplicates and sort
    vrp_files = sorted(list(set(vrp_files)))
    return vrp_files


def save_results(results: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save evaluation results to CSV file.
    
    Args:
        results: List of evaluation result dictionaries
        output_path: Path to output CSV file
    """
    if not results:
        logger.warning("No results to save")
        return
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    columns = ['file_name', 'type', 'dimension', 'total_distance', 
               'optimal_distance', 'relative_error_percent', 'inference_time']
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)
    
    logger.info(f"Results saved to: {output_path}")


def main():
    """Main function to run performance evaluation."""
    parser = argparse.ArgumentParser(
        description='Performance Evaluation for CVRP Jaya Algorithm'
    )
    parser.add_argument(
        'output_path',
        type=str,
        help='Path to output CSV file for storing results'
    )
    parser.add_argument(
        '--vrp_dir',
        type=str,
        default='./samples',
        help='Directory containing .vrp files (default: ./instances)'
    )
    parser.add_argument(
        '--vrp_files',
        type=str,
        nargs='+',
        help='Specific .vrp files to evaluate (overrides --vrp_dir)'
    )
    parser.add_argument(
        '--pop_size',
        type=int,
        default=30,
        help='Population size for Jaya algorithm (default: 30)'
    )
    parser.add_argument(
        '--max_iter',
        type=int,
        default=200,
        help='Maximum iterations for Jaya algorithm (default: 200)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("CVRP Jaya Algorithm Performance Evaluation")
    logger.info("=" * 60)
    logger.info(f"Output path: {args.output_path}")
    logger.info(f"Population size: {args.pop_size}")
    logger.info(f"Max iterations: {args.max_iter}")
    
    # Get list of VRP files to evaluate
    if args.vrp_files:
        vrp_files = args.vrp_files
        logger.info(f"Evaluating {len(vrp_files)} specified files")
    else:
        vrp_files = find_vrp_files(args.vrp_dir)
        logger.info(f"Found {len(vrp_files)} .vrp files in {args.vrp_dir}")
    
    if not vrp_files:
        logger.error("No .vrp files found to evaluate")
        sys.exit(1)
    
    # Evaluate each instance
    results = []
    total_files = len(vrp_files)
    
    for idx, vrp_file in enumerate(vrp_files, 1):
        logger.info(f"\n[{idx}/{total_files}] Processing file...")
        
        try:
            result = evaluate_instance(
                vrp_file,
                pop_size=args.pop_size,
                max_iter=args.max_iter
            )
            results.append(result)
            logger.info(f"  ✓ Successfully evaluated")
            
        except FileNotFoundError:
            logger.error(f"  ✗ File not found: {vrp_file}")
            continue
        except Exception as e:
            logger.error(f"  ✗ Error processing {vrp_file}: {str(e)}")
            continue
    
    # Save results
    logger.info("\n" + "=" * 60)
    logger.info("Saving results...")
    save_results(results, args.output_path)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total files processed: {len(results)}/{total_files}")
    
    if results:
        # Group by type
        types = {}
        for r in results:
            t = r['type']
            if t not in types:
                types[t] = []
            types[t].append(r)
        
        logger.info(f"\nResults by task type:")
        for task_type, type_results in sorted(types.items()):
            avg_time = sum(r['inference_time'] for r in type_results) / len(type_results)
            avg_dim = sum(r['dimension'] for r in type_results) / len(type_results)
            
            # Calculate average relative error only for results with valid values
            error_values = [r['relative_error_percent'] for r in type_results if r['relative_error_percent'] != '']
            avg_error = sum(error_values) / len(error_values) if error_values else None
            
            logger.info(f"  Type {task_type}:")
            logger.info(f"    Count: {len(type_results)}")
            logger.info(f"    Avg dimension: {avg_dim:.1f}")
            logger.info(f"    Avg inference time: {avg_time:.2f}s")
            if avg_error is not None:
                logger.info(f"    Avg Relative Error (%): {avg_error:.2f}%")
    
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()