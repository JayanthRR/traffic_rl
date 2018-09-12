# traffic_rl

### Procedure to run simulations and generate results
#### Run the simulation on server (or a local machine).
1) Create a virtual environment with python3. All the simulations must run inside the virtual environment. 
2) Then run the following commands
`pip install -r requirements.txt`
3) Go to `main.py` and do any necessary changes to `config` dictionary. 
4) Run `main.py`. The logs will be saved with the timestamp in `logs\` folder. 
5) Each logfile is a folder named after the timestamp at which the simulation was run. The folder consists of the following files.
* `config.p` : Saves the configuration used for the simulations
* `env.p` : Saves the `TrafficEnv` computed using the config and used in the simulations
* `logs.p` : Contains the testing log details of all the algorithms in `test_config`
* `<algorithm>.p` : Saves the parameters learnt after training using `<algoritm>`
* `_<algorithm>_log.p` : Saves the training details of the `<algorithm>`

#### Copy the logs onto a local machine (since servers usually do not have displays to visualize the plots).
1) If you are already on a local machine, there is no need for this step.
2) zip the files as follows (from the root directory `traffic_rl\`
`zip logfile.zip -r logs\*`
3) scp the zip file from server onto the local machine
4) Assuming that the zip file is placed in the local machine's project root folder `traffic_rl\`, run the following command
`unzip logfile.zip`

#### Visualization
1) Run `plots.py`.
2) Check for each logfile's individual folder for that simulation's training and testing related plots
3) An aggregate plot containing the testing logs of different configurations is generated in the root folder.


