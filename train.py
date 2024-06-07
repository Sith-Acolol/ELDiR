import os
import subprocess
import numpy as np

# Define the base directory and the generation to resume from
base_dir = 'run-out'
resume_generation = 35

# Define the number of iterations and random seeds
iterations = 10
seeds = np.random.randint(0, np.iinfo(np.int32).max, size=iterations)

for i, seed in enumerate(seeds):
    print(f"Running iteration {i + 1} with seed {seed}")
    
    # Define the checkpoint directory for this iteration
    ckptdir = os.path.join(base_dir, str(resume_generation))
    
    # Define a unique output directory for each iteration
    iteration_outdir = os.path.join(base_dir, f'iteration_{i + 1}')
    os.makedirs(iteration_outdir, exist_ok=True)
    
    # Check for args.txt in the run-out directory
    args_path = os.path.join(base_dir, 'args.txt')
    if not os.path.exists(args_path):
        raise FileNotFoundError(f"args.txt not found in {base_dir}")
    
    # Read the args.txt file
    with open(args_path, 'r') as f:
        arg_options = eval(f.read())
    
    # Update the options with the new seed, max_generations, and new outdir
    arg_options['seed'] = seed
    arg_options['outdir'] = iteration_outdir

    # Write the updated args.txt to the specific generation directory
    with open(os.path.join(iteration_outdir, 'args.txt'), 'w') as f:
        f.write(str(arg_options) + '\n')

    # Run the command to resume training with the new outdir
    command = f"python run.py --ckptdir {ckptdir}"
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Error occurred during iteration {i + 1}")
    else:
        print(f"Iteration {i + 1} completed successfully")

print("All iterations completed.")