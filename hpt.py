import numpy as np
from typing import List, Dict, Any
import time
import os
import json
import pandas as pd
import warnings
from pathlib import Path
import argparse
from jaya import Jaya

warnings.filterwarnings('ignore')

class JayaHyperparameterTuner:

    def __init__(
        self,
        pop_size_range: List[int] = [10, 20, 30, 50, 75, 100],
        max_iter: int = 200,
        verbose: bool = True
    ):
        self.pop_size_range = pop_size_range
        self.max_iter = max_iter
        self.verbose = verbose

        self.tuning_results: Dict[str, Dict] = {}
        self.best_params: Dict[str, Dict] = {}

    def _evaluate_config(self, vrp_file: Path, pop_size: int) -> Dict[str, Any]:
        start_time = time.time()
        solver = Jaya(pop_size=pop_size, max_iter=self.max_iter)

        try:
            result = solver.predict(str(vrp_file))
            elapsed = time.time() - start_time
            gap = solver.get_gap()

            return {
                'pop_size': pop_size,
                'total_distance': result['total_distance'],
                'num_vehicles': result['num_vehicles'],
                'gap': gap if gap is not None else float('inf'),
                'elapsed_time': elapsed,
                'optimal_value': result.get('optimal_value'),
                'success': True,
                'error': None
            }
        except Exception as e:
            return {
                'pop_size': pop_size,
                'total_distance': float('inf'),
                'num_vehicles': float('inf'),
                'gap': float('inf'),
                'elapsed_time': time.time() - start_time,
                'optimal_value': None,
                'success': False,
                'error': str(e)
            }

    def tune_single_sample(self, vrp_file: Path, sample_name: str = None) -> Dict[str, Any]:
        if sample_name is None:
            sample_name = vrp_file.stem

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🔍 Tuning (single run) for: {sample_name}")
            print(f"{'='*60}")
            print(f"  pop_size candidates: {self.pop_size_range}")
            print(f"  max_iter: {self.max_iter} (fixed)")
            print()

        results = []
        summary = {}

        for pop_size in self.pop_size_range:
            if self.verbose:
                print(f"  Testing pop_size={pop_size}...", end=' ')

            res = self._evaluate_config(vrp_file, pop_size)
            results.append(res)

            if res['success']:
                summary[pop_size] = {
                    'distance': res['total_distance'],
                    'gap': res['gap'],
                    'time': res['elapsed_time'],
                    'num_vehicles': res['num_vehicles'],
                    'success': True
                }
                if self.verbose:
                    print(f" distance={res['total_distance']:.2f}, gap={res['gap']:.2f}%, time={res['elapsed_time']:.1f}s")
            else:
                summary[pop_size] = {
                    'distance': float('inf'),
                    'gap': float('inf'),
                    'time': res['elapsed_time'],
                    'num_vehicles': float('inf'),
                    'success': False
                }
                if self.verbose:
                    print(f" FAILED")

        valid_summary = {ps: s for ps, s in summary.items() if s['success']}
        if valid_summary:
            best_pop_size = min(valid_summary.keys(), key=lambda ps: valid_summary[ps]['distance'])
            best_params = {
                'pop_size': best_pop_size,
                'max_iter': self.max_iter,
                'distance': summary[best_pop_size]['distance'],
                'gap': summary[best_pop_size]['gap'],
                'time': summary[best_pop_size]['time']
            }
        else:
            best_pop_size = self.pop_size_range[0]
            best_params = {
                'pop_size': best_pop_size,
                'max_iter': self.max_iter,
                'distance': float('inf'),
                'gap': float('inf'),
                'time': float('inf')
            }
            if self.verbose:
                print("\n⚠️  No successful runs for any configuration!")

        self.tuning_results[sample_name] = {
            'all_results': results,
            'summary': summary,
            'best_params': best_params
        }
        self.best_params[sample_name] = best_params

        if self.verbose:
            print(f"\n✅ Best for {sample_name}: pop_size={best_pop_size}, distance={best_params['distance']:.2f}, gap={best_params['gap']:.2f}%")

        return {
            'sample_name': sample_name,
            'best_params': best_params,
            'summary': summary,
            'all_results': results
        }

    def tune_multiple_samples(self, vrp_files: List[Path]) -> Dict[str, Dict]:
        all_results = {}
        for idx, f in enumerate(vrp_files):
            name = f.stem
            if self.verbose:
                print(f"\n[{idx+1}/{len(vrp_files)}] Processing {name}...")
            res = self.tune_single_sample(f, name)
            all_results[name] = res
        return all_results

    def get_results_dataframe(self) -> pd.DataFrame:
        records = []
        for name, params in self.best_params.items():
            records.append({
                'sample_name': name,
                'best_pop_size': params['pop_size'],
                'max_iter': params['max_iter'],
                'distance': params['distance'],
                'gap': params['gap'],
                'time': params['time']
            })
        return pd.DataFrame(records)

    def get_detailed_results_dataframe(self) -> pd.DataFrame:
        records = []
        for sample_name, data in self.tuning_results.items():
            for pop_size, summ in data['summary'].items():
                records.append({
                    'sample_name': sample_name,
                    'pop_size': pop_size,
                    'max_iter': self.max_iter,
                    'distance': summ['distance'],
                    'gap': summ['gap'],
                    'time': summ['time'],
                    'num_vehicles': summ['num_vehicles'],
                    'success': summ['success']
                })
        return pd.DataFrame(records)

    def save_results(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / 'tuning_results.json'
        csv_path = output_dir / 'tuning_results.csv'

        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj

        out = {
            'best_params': convert(self.best_params),
            'tuning_config': {
                'pop_size_range': self.pop_size_range,
                'max_iter': self.max_iter
            }
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2)

        detailed_df = self.get_detailed_results_dataframe()
        detailed_df.to_csv(csv_path, index=False)

        if self.verbose:
            print(f"\n💾 Results saved to {output_dir}")

    def load_results(self, input_path: Path) -> None:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.best_params = data['best_params']
        if self.verbose:
            print(f"📂 Loaded results from {input_path}")


def run_grid_search_tuning(
    input_dir: Path,
    output_dir: Path,
    pop_size_range: List[int] = [10, 20, 30, 50, 75, 100],
    max_iter: int = 200,
    verbose: bool = True
) -> Dict[str, Dict]:
    vrp_files = list(input_dir.glob('*.vrp'))
    if not vrp_files:
        print(f"No .vrp files found in {input_dir}")
        return {}

    tuner = JayaHyperparameterTuner(
        pop_size_range=pop_size_range,
        max_iter=max_iter,
        verbose=verbose
    )
    tuner.tune_multiple_samples(vrp_files)
    tuner.save_results(output_dir)

    if verbose:
        print("\n" + "="*60)
        print("📊 FINAL SUMMARY (single run per config)")
        print("="*60)
        df = tuner.get_results_dataframe()
        print(df.to_string(index=False))
        print("\nPopulation size distribution:")
        print(df['best_pop_size'].value_counts().sort_index())
    return tuner.best_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str, help='Directory containing .vrp files')
    parser.add_argument('output_dir', type=str, help='Directory to save results')
    parser.add_argument('--pop_sizes', nargs='+', type=int, default=[200,400,500,1000],
                        help='List of population sizes to try')
    parser.add_argument('--max_iter', type=int, default=100, help='Maximum iterations (fixed)')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress verbose output')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    run_grid_search_tuning(
        input_dir=input_dir,
        output_dir=output_dir,
        pop_size_range=args.pop_sizes,
        max_iter=args.max_iter,
        verbose=not args.quiet
    )