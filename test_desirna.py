import subprocess
import os

target_structure = "((((......))))"
seq_restr = "N" * len(target_structure)
input_content = f">name\nDesign\n>seq_restr\n{seq_restr}\n>sec_struct\n{target_structure}\n"

with open("desirna_test_input.txt", "w", encoding="utf-8") as f:
    f.write(input_content)

print("Running DesiRNA...")
subprocess.run([r".\protein_sim\Scripts\python.exe", "DesiRNA/DesiRNA.py", "-f", "desirna_test_input.txt", "-t", "5", "-r", "5"])

print("\nListing output directories:")
for root, dirs, files in os.walk("."):
    for dir in dirs:
        if "desirna_test_input" in dir:
            print(os.path.join(root, dir))
            for f in os.listdir(os.path.join(root, dir)):
                print(f"  - {f}")
