import csv
# from phase15_full_evaluation import run_full_evaluation
# pyrefly: ignore [missing-import]
from phase17_extended import evaluate_extended_qubo
def load_eterna100(filepath):
    structures = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        # Skip the header row
        next(reader)
        for row in reader:
            if len(row) > 3:
                # Column 3 (0-indexed) is 'Secondary Structure V2'
                struct = row[3].strip()
                if struct and len(struct) <=60:
                    structures.append(struct)
    return structures

if __name__ == "__main__":
    # Path to your Eterna100 dataset
    tsv_path = r"d:\Academic UOP\Internship\simulation\Implementation\NN - Copy\Structures\eterna100_puzzles.tsv.txt"
    
    print("Loading Eterna100 dataset...")
    eterna_structures = load_eterna100(tsv_path)
    print(f"Successfully loaded {len(eterna_structures)} puzzles.")

    # Eterna100 contains massive structures (up to 400 bases). 
    # Running all 100 at once with huge pool sizes will take days.
    # We will test the first 5 easiest puzzles first to ensure everything works!
    test_batch = eterna_structures[:30]
    
    # print(f"\nRunning evaluation on {len(test_batch)} Eterna puzzles...")
    # run_full_evaluation(
    #     target_structures=test_batch,
    #     num_sa_reads=1000, 
    #     min_random_assignments=10,
    #     pool_size=1000,                 # Reduced from 1000 for speed on large structures
    #     variations_per_assignment=10, 
    #     num_corr_samples=500           # Reduced from 500 for speed
    csv_file = "results/phase16_extended_benchmark.csv"
    import os
    os.makedirs("results", exist_ok=True)
    
    print(f"\nRunning Extended QUBO benchmark on {len(test_batch)} Eterna puzzles...")
    print(f"Saving results to: {csv_file}\n")
    
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Target Structure", "Length", "Total Valid Reads", "Successful Folds", "Success Rate (%)"])
        
        for i, target in enumerate(test_batch):
            print(f"\n--- Benchmark Puzzle {i+1}/{len(test_batch)} ---")
            success_count, total_reads, top_10 = evaluate_extended_qubo(target, num_reads=10000)
            rate = (success_count / 10.0) * 100
            
            writer.writerow([target, len(target), total_reads, success_count, f"{rate:.1f}"])
            f.flush()  # Save instantly so progress isn't lost if stopped early
            
    print(f"\nBenchmark Complete! Results saved to {csv_file}")