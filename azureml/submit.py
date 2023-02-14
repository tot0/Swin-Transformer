# %%
from azure.ai.ml import MLClient, command, Input, Output, MpiDistribution, PyTorchDistribution
from azure.ai.ml.entities import JobService
from azure.identity import DefaultAzureCredential

# cd kernels/window_process/ && \
# python setup.py install && cd ../../ && \
# PYTHONUNBUFFERED=1 python main.py

input_mode = 'download'
cache_mode = 'full'

settings = {
    "input_mode": ['download', 'ro_mount'],
    "DATASET_MOUNT_BLOCK_BASED_CACHE_ENABLED": ['true', 'false'], # _blockbase
    "cache": ['full', 'part', 'no'],
}

imagenet_smallfiles = Input(
    type="uri_folder",
    mode=input_mode,
    path="azureml://datastores/imagenet/paths/data/imagenet/",
)
imagenet_smallfiles_subset = Input(
    type="mltable",
    mode='eval_download',
    path="azureml:imagenet_7percent_v1:1",
)

def submit_pytorch_job(ml_client: MLClient, compute_name='a100-900ram-low'):
    command_job = command(
        code="../",
        command="bash launch.bash --cfg ${{inputs.config_file}} --data-path ${{inputs.imagenet_data}} --batch-size ${{inputs.batch_size}} --output ${{outputs.trained_model}} --cache-mode ${{inputs.cache_mode}} --enable_netmon ${{inputs.enable_netmon}} --opts TRAIN.EPOCHS ${{inputs.num_epochs}} TRAIN.WARMUP_EPOCHS ${{inputs.num_warmup_epochs}} DATA.NUM_WORKERS ${{inputs.num_workers}} PROFILING ${{inputs.enable_profiling}} TENSORBOARD_OUTPUT_DIR $AZUREML_CR_EXECUTION_WORKING_DIR_PATH LOG_OUTPUT_DIR $AZUREML_CR_EXECUTION_WORKING_DIR_PATH/user_logs",
        # torchrun --nproc_per_node ${{inputs.num_gpus}} --master_port 12345
        environment="acpt-pytorch-113-py38-cuda117-gpu-swin:4",
        environment_variables={
            'NCCL_DEBUG': 'INFO',
            'DATASET_MOUNT_BLOCK_BASED_CACHE_ENABLED': 'true',
            #'DATASET_MOUNT_MEMORY_CACHE_SIZE': str(200*1024*1024*1024),
            '_AZUREML_CR_APPLICATION_INSIGHTS_LOG_LEVEL_OVERRIDE': 'debug',
        },
        inputs={
            # data inputs
            "imagenet_data": imagenet_smallfiles_subset,
            "config_file": "configs/swinv2/swinv2_small_patch4_window8_256.yaml", # swinv2_base_patch4_window8_256.yaml
            # data loading
            "batch_size": 128,
            "cache_mode": cache_mode,
            "num_workers": 8, # Default from config.py
            #"prefetch_factor": 4,
            "num_gpus": 8,
            # model
            # training
            "num_warmup_epochs": 2,
            "num_epochs": 10,
            # profiling
            "enable_profiling": False,
            "enable_netmon": True
        },
        outputs={
            "checkpoints": Output(type="uri_folder"),
            "trained_model": Output(type="uri_folder")
        },
        compute=compute_name,
        distribution=PyTorchDistribution(process_count_per_instance=8),# MpiDistribution(process_count_per_instance=1),
        instance_count=2,
        shm_size="128g",
        services={
            'tensorboard': JobService(job_service_type='tensor_board', properties={'logDir': 'tensorboard_logs'}),
        #     'jupyterlab': JobService(job_service_type='jupyter_lab'),
        #     'ssh': JobService(job_service_type='ssh'),
        #     'vscode': JobService(job_service_type='vs_code')
        }
    )

    returned_job = ml_client.jobs.create_or_update(command_job, tags={'input_mode': input_mode, 'cache_mode': cache_mode}, experiment_name='swinv2_imagenet1k')
    print(returned_job.studio_url)


# %%
def __main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--sub-id", type=str, default='') # 
    parser.add_argument("--resource-group", type=str, default='lupickup-dev')
    parser.add_argument("--workspace-name", type=str, default='lupickup-test-eastus')
    parser.add_argument("--compute-name", type=str, default='lupickup-8v100-low') # 'a100-900ram-low')
    args = parser.parse_args()

    ml_client = MLClient(
        DefaultAzureCredential(), subscription_id=args.sub_id, resource_group_name=args.resource_group, workspace_name=args.workspace_name
    )

    submit_pytorch_job(ml_client, compute_name=args.compute_name)


if __name__ == '__main__':
    __main()