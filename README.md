# Cityflows Data Model

## Introduction

This repository contains the Cityflows core model, along with wrapping code and Docker setup to deploy it and make it available as a service. This readme describes how to get the model up and running. More information on the motivation and technical setup of the model can be found on the [model explanation](./model_explanation.md) page.

## Conda environments

| file                     | description                                                                                              | target audience | when to use                                                                            | remark                                                              |
| ------------------------ | -------------------------------------------------------------------------------------------------------- | --------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `environment-docker.yml` | environment for the Docker image, stripped down of all non-essential dependencies.                       | docker image    | automatic activation, when [Running the service locally](#Running-the-service-locally) |                                                                     |
| `environment-local.yml`  | environment for local development/testing, including all dependencies for Jupyter notebooks for example. | data scientists | manual activation, before [Running the model locally](#Running-the-model-locally)      | this works on MacOS 11.2 and Ubuntu 20.4, not tested on Windows yet |

### Create the environment for data scientists

```
conda env create -f environment-local.yml
```

### Activate the environment for data scientists

```
conda activate cityflows-model
```

### Update the environment for data scientists

```
conda install <package>
conda env export --from-history > environment-local.yml
```

### Update the environment for Docker image

When adding a new package for the deployed model (not for testing purposes or notebooks experiments), please update the `environment-docker.yml` file by manually adding a line for the package and its version (do not include the build number).

## Running the model locally

> Make sure your activated conda environment is `cityflows-model`

From the root of the repository (don't `cd src/model`)

```
python -m src.model.Main
```

> Other scripts should be executed in a similar way, with folders separated by `.` instead of `/` after the `-m` argument. It's very important to use the `-m` argument to tell python to execute `Main` as a module, this way relative imports work as expected. Here's a [good read](https://stackoverflow.com/questions/14132789/relative-imports-for-the-billionth-time) about relative imports in Python.

### Data requirements

For a local run of the model you of course need data. At the bottom in the main part of the `src/model/Main.py` script you define an input- and outputfolder. In the inputfolder the following files need to be present:

- `street_segments.csv`
- `intersections.csv`
- `all_data.csv`
- `ml_weights.csv` (optional)

The first 2 files are the results of the [roadcutter procedure](https://github.com/imec-int/cityflows-road-cutter). This procedure uses the flemish roadregister and the `all_data.csv`.

The `all_data.csv` is the result of collection/scraping for every datasource followed by their respective projection into the correct format (see `src/handling_scripts`). The final step is to use the `src/tooling/merge/main.py` script to merge the different datasources in a single file. 

Finally the fourth and final file is the `ml_weights.csv` which is the output of the [machine learning solution](https://github.com/imec-int/cityflows-ml-weights). This contains a weight of how likely it is to find a specific modality in a specific street. This file is optional if it is not present all the values or weights will be set to a default value of 1. So we default to a uniform distribution of the modalities.
 
## Running the service locally

### Setup

To run the service locally, you need to emulate the circumstances in the k8s cluster.

Follow these steps to do so:

1.  Create the directory for the secrets `mount/kvmnt` and add a file called AZURE-CITYFLOWS-STORAGE-KEY (you can also choose to set the env variable `AZURE_CITYFLOWS_STORAGE_KEY` and skip step 2)
2.  In that file, put the account access key for the Azure blob storage account you wish to read model inputs from and write model outputs to
3.  If you plan to use the gurobi solver, make sure your machine has a gurobi key with a corresponding license.

4.  (Optional) Override configuration by creating file `mount/config.yaml` that follows the schema of `src/service_utils/config_default.yaml`. You don't need all fields, only specify the fields that you wish to override. For example:

```
azure:
  input:
    blob_connection_string: "DefaultEndpointsProtocol=https;AccountName=digdevelopmentcityflows;AccountKey={0};EndpointSuffix=core.windows.net"
    container_name: "model-input"
  output:
    blob_connection_string: "DefaultEndpointsProtocol=https;AccountName=digdevelopmentcityflows;AccountKey={0};EndpointSuffix=core.windows.net"
    container_name: "model-output-debug"
```

> Tip: you can disable various steps of the service execution in `mount/config.yaml` file, see `src/service_utils/config_default.yaml` for more information

### How to start up the service

Run `./build.sh && ./up.sh`.

You should be able to connect to the kafka instance inside docker by listening to `localhost:9093`.
You can use kaf CLI to consume and produce messages:

1. config kaf to connect to the correct cluster
2. `kaf config select-cluster`
3. `kaf topics`

### Model run execution

The model has 2 modes to operate: it can either run as a `service` or as a `cronjob`. You can change the mode by changing the `server.enabled` flag (e.g. in `mount/config.yaml`)

#### Service mode

In this mode it will start a permanent service that waits for commands on kafka before it starts a model run.
By sending a message to kafka using

```
cat ./service_command_messages/<file.json> | tr '\n' ' ' | kaf produce cmd.cityflows.model.1 -k start
```

you will invoke the model using the files listed in the file `<file.json>`. This is convenient to locally reproduce a bug that ocurred on the deployed service. You can also upload input files of your choice to blob storage and trigger a model run.

#### Cronjob mode

In this mode it will automatically start the model run and will download the most recent files from blob storage. The model will automatically publish and exit when completed. This cronjob mode is still maintained but it will probably be deprecated.

## Execute the model as batches in Kubernetes

When computing for a large dataset, we don't want to run those heavy computations on our development machines, limiting ourselves in the kind of paralell tasks we can work on because the machine is pretty busy computing, which makes resources somewhat scarce. Furthermore, we don't want those computations to run for countless days, hence we want more processing speed. In order to allow that, we worked on a way to easily deploy multiple [Jobs](https://kubernetes.io/docs/concepts/workloads/controllers/job/) on Kubernetes that will split the work between themselves, resulting in horizontal scaling of the CityFlows model.

### Blob storage structure

To facilitate this, we will make heavy use of an Azure Blob Storage. It will hold the input files/batches that the Kubernetes Jobs will automatically download to perform their computations. Upon successful completion of a batch, the Jobs will automatically upload the results of the computation on this blob storage, in a dedicated folder.

It's important to adhere to the following structure (where `<BLOBS_DIR>` can be any path in the blob storage):

```
<BLOBS_DIR>
└───input
│   │   intersections.csv
│   │   street_segments.csv
│   └───counts
│       │   cropland_2020_01_01.csv
│       │   cropland_2020_01_02.csv
|       |   ...
|       |   cropland_2020_12_31.csv
```

The same `intersections.csv` and `street_segments.csv` files will be shared and used amongst all jobs/batches. Then, each file in the `counts` folder is what we call a batch. Batches contain counts data for non-overlapping time periods that can run independently. In the example above, we split the Cropland counts data of a whole year into 366 files. We could have split differently, into 52 weeks, 12 monthes, 3 quarters, etc. We recommend not going smaller than a day though.

> Note that splitting into 366 files does not mean that we will spawn 366 Jobs, that will be explained in the following section.

Upon successful completion, the blob storage will have the following structure

```
<BLOBS_DIR>
└───input
│   │   intersections.csv
│   │   street_segments.csv
│   └───counts
│       │   cropland_2020_01_01.csv
│       │   cropland_2020_01_02.csv
|       |   ...
|       |   cropland_2020_12_31.csv
└───output
│   │   street.csv
│   └───densities
│       │   cropland_2020_01_01.csv
│       │   cropland_2020_01_02.csv
|       |   ...
|       |   cropland_2020_12_31.csv
```

The `streets.csv` file maps a `street_object_id` to its corresponding street geometry and potentially other information. It became necessary as the csv files in the `output/densities` were stripped of the street geometries in order to keep the file sizes minimal. This file can easily be joined/merged back with the results in Pandas if you care about conducting a spatial analysis of the results.

The files in `output/densities` then contain the results of the computation for batches of the same name. Notice the 1-to-1 match between files in `input/counts` and `output/densities`.

### Configure the Kubernetes context

To run those batched computations, we will make use of a dedicated Kubernetes cluster, called `aks-cityflows-batch`. On the first time, you will need to make this context available in your `kubectl` config. In order to do that, execute

```
az login
az account set --subscription EDiT-CityFlows
az aks get-credentials -g digital-twin-cityflows -n aks-cityflows-batch
```

Next times, you will only need to make sure this context is active by executing

```
kubectl config use-context aks-cityflows-batch
```

### Configuration

The Job containers will take their configuration from a ConfigMap called `batch-job-config` that already exists in the Kubernetes cluster. This ConfigMap contains the credentials for the containers to access the Blob Storage.

### How to run

In order to run a batched computation, you need to configure it via the `batch_execution/create_batch_jobs.sh` script. 2 variables need to be configured:

- `BLOBS_DIR`: the blob path containing the input files (see previous section for explanation)
- `COUNTS_PREFIXES_LIST`: This is where we specify which batches are handled by which Job. It is an array that contains as many items as there will be jobs. Each item of the array is a string that gives one or multiple blob patterns for the batches it will handle. In order to know how to format those blob patterns, be aware that under the hood, blob patterns are passed to the `name_starts_with` named argument of [this function](https://docs.microsoft.com/en-us/python/api/azure-storage-blob/azure.storage.blob.containerclient?view=azure-python#list-blobs-name-starts-with-none--include-none----kwargs-).

> Example: If a job has an item in the `COUNTS_PREFIXES_LIST` array equal to `"cropland_2020_01_01 cropland_2020_02_0 cropland_2020_03"`, then it means that it will handle: the first day of January, the first 10 days of February and the whole month of March, resulting in 42 batches.

Jobs will run in parallel, but each job will work sequentially on the batches it has to handle, chronologically.

Once you configured the `batch_execution/create_batch_jobs.sh` to your needs, run it. It will not spawn the Jobs directly, it will just create manifests for them. Hence, don't be scared to run it. The manifests will be created in a folder called `batch-jobs`. Feel free to inspect them and spot potential mistakes. If everything looks in order, then spawn them by running:

```
cd batch_execution
```

```
kubectl create -f batch-jobs
```

Use `k9s` or

```
kubectl get -f batch-jobs
```

to get information on their execution.

Cancel them or clean them after a successful completion by executing

```
kubectl delete -f batch-jobs
```

> It is also possible to target a single job by running `kubectl <create|get|delete> -f batch-jobs/job-0.yaml`

## Tooling

In this section we are going to list and describe tooling scripts that could make the developer's life easier.

| **Location**                                   | **Description**                                                                                                                                               | **Usage Example**                                                                                                        |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| manage_data.sh                                 | Utility script for downloading data files hosted on the CityFlows blob storage                                                                                | ./manage_data.sh download mobiele_stad/learning_cycle_1                                                                  |
| src/tooling/merge/main.py                      | Script for merging several count files coming from different sources. Optional spatial and temporal filtering is also supported                               | Have a look in execute_LC.sh                                                                                             |
| src/tooling/modelling_zone/Main.py             | Script for creating a modelling zone based on a boundary zone and exclusion zones                                                                             | python -m src.tooling.modelling_zone.Main                                                                                |
| src/tooling/output_checks/check_alphas.py      | Script for assessing the output of a model run, based on the alphas                                                                                           | python -m src.tooling.output_checks.check_alphas data/managed_data_files/mobiele_stad/learning_cycle_3/output/alphas.csv |
| src/tooling/output_checks/investigate_cells.py | Script for comparing the counts computed by the model to the input counts, on a datasource cell basis                                                         | python -m src.tooling.output_checks.investigate_cells --data_source cropland --data_source_cells 153581 --modality all   |
| src/tooling/shapes/extract.py                  | Script extracting the unique shapes out of all_counts.csv files                                                                                               | python -m src.tooling.shapes.extract --input_file_path all_data.csv --output_file_path all_shapes.csv                    |
| src/tooling/straatvinken_transformation/\*     | Scripts for transforming the Straatvinken raw dataset into the format expected by the validation scripts. More info in the Readme file located in that folder |                                                                                                                          |
| src/tooling/test_set/\*                        | Scripts for creating test sets (subsets of existing files)                                                                                                    |                                                                                                                          |
| src/tooling/validation/score.py                | Script to compute a validation score                                                                                                                          | python -m src.tooling.validation.score                                                                                   |
| src/tooling/validation/visualise.py            | Script to visualise a validation session data file                                                                                                            | python -m src.tooling.validation.visualise                                                                               |

# Connected repositories in the Cityflows ecosystem
- the road cutter procedure: [repository](https://github.com/imec-int/cityflows-road-cutter)
- weighted distribution: [repository](https://github.com/imec-int/cityflows-ml-weights)