This repository contains the python code for training neural networks on simulated radar response to ideal point reflectors for the purpose of radar signal super-resolution. 

This work was done as part of Mumin Jin's MEng Thesis titled *Machine Learning Methods for Super-Resolution inSparse Sensor Arrays*

Requirements

Data generation
```
python data_generation.py --mode <mode> --directory <directory> --sample_type <sample_type> --num_scenes <num_scenes> --mix 0
```
The result is having <num_scenes> of samples in the directory cloud_data/<directory>/<mode>

Check data
```
python data_manage.py --data_directory cloud_data <directory> <mode> --nums_show <nums_show>
```

Train network
```
python train.py --arg_file <arg_file>
```
where <arg_file>.json is a file in the folder train_args
