# Pipe Swarm Obstacle Avoidance Simulation

This simulation is a replica of `sim2sim/drone_airsim/swarm` with the
central wall opening replaced at runtime by a lattice of cylindrical pipe
obstacles. The AirSim package is unchanged; `eval.py` removes the original
central wall objects after reset, spawns the pipes through the AirSim Python
API, and records the pipe layout in each experiment directory.

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

To evaluate the replicated scene with the original swarm checkpoint from the
repository root:

```bash
conda run -n diffphysdrone python eval.py --resume /home/ember/GitHub/diffphysdrone_pipeline_inspection/sim2sim/drone_airsim/swarm/swarm.pth --target_speed 2.5
```

During execution, the system will output the task completion time for each agent, along with any collision information. The expected processing speed is 3.75 it/s (15Hz with a clock speed setting of 0.25), and the simulation should complete in approximately 40 seconds.

We provide the script `batch_test.sh` for conducting 10 sequential runs of the simulation.

## Viewing Test Results and Videos

Each run writes `pipe_scene.json`, `traj_history.json`, `wind_trace.jsonl`,
`wind_summary.json`, `depth.mp4`, and `log` under `./exps_<target_speed>/`.
To view a comprehensive log of all test results, use the following command:

```bash
tail exps_*/*/log
```

This command will display the latest entries from the log files of all experiments, allowing you to review the outcomes of each simulation run.
