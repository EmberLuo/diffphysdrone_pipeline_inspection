# Communication-free Swarm Position Swapping Simulation

## Environment Setup

The simulation has been validated under the following configuration:

- CPU: AMD Ryzen 2700X
- Memory: 64GB DDR4 at 2800MHz
- GPU: NVIDIA RTX 4060ti
- Operating System: Ubuntu 20.04
- Python Version: 3.8.18
- PyTorch Version: 2.1.2

## Running the Simulation

### Starting the AirSim Simulator

Execute the following command to launch the AirSim simulator:

```bash
./LinuxNoEditor/Blocks.sh -ResX=896 -ResY=504 -windowed -WinX=512 -WinY=304 -settings=$PWD/settings.json
```

Upon successful launch, a window will appear displaying the first-person view from one of the drones.

### Executing the Swarm Planner

You may need to install the airsim package via `pip install airsim`. The expected installation time is around 1 minute.
Open a new terminal window and initiate our swarm planner by running:

```bash
python eval.py --resume swarm.pth --target_speed 2.5
```

During execution, the system will output the task completion time for each agent, along with any collision information. The expected processing speed is 3.75 it/s (15Hz with a clock speed setting of 0.25), and the simulation should complete in approximately 40 seconds.

We provide the script `batch_test.sh` for conducting 10 sequential runs of the simulation.

## Viewing Test Results and Videos

Example test results with videos demonstrations are provided in `./exps_2.5`, such as `exps_2.5/20240225_145242/20240225_145242.mp4`. To view a comprehensive log of all test results, use the following command:

```bash
tail exps_*/*/log
```

This command will display the latest entries from the log files of all experiments, allowing you to review the outcomes of each simulation run.
