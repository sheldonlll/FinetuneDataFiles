# LMFlow:  
在data目录下新建MIT文件夹  
在MIT文件夹下新建train_modify文件夹，并将finetune数据文件放入  
在MIT文件夹下新建test文件夹，并将evaluate数据文件放入  

[finetune数据：traindata_LMFlow.json](https://github.com/sheldonlll/FinetuneDataFiles/blob/main/traindata_LMFlow.json)  

[evaluate数据：testdata_LMFlow.json](https://github.com/sheldonlll/FinetuneDataFiles/blob/main/testdata_LMFlow.json)  
 
将scripts目录中的run_finetune.sh， run_evaluate.sh文件替换  
切换到scripts目录中，执行命令：  

[1. ./run_finetune.sh](https://github.com/sheldonlll/FinetuneDataFiles/blob/main/run_finetune.sh#L16)  

[2. ./run_evaluate.sh](https://github.com/sheldonlll/FinetuneDataFiles/blob/main/run_evaluation.sh#L5)  

  
# ChatGLM6B:  
clone the repo, skip large files: 
`GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/THUDM/chatglm-6b`  

Download large files from Tsinghua Cloud  

`git clone git@github.com:chenyifanthu/THU-Cloud-Downloader.git`
`cd THU-Cloud-Downloader`
`pip install argparse requests tqdm`  
`python main.py --link https://cloud.tsinghua.edu.cn/d/fb9f16d6dc8f482596c2/ --save ../chatglm-6b/`  

Clone Source Code  

`git clone git@github.com:THUDM/ChatGLM-6B.git`

ChatGLM-6B文件夹与chatglm-6b文件夹在同一级目录下
 
在ChatGLM-6B/ptuning/AdvertiseGen/目录中  
将train数据文件放入  
将evaluate数据文件放入

[train数据：traindata_ChatGLM6B.json](https://github.com/sheldonlll/FinetuneDataFiles/blob/main/traindata_ChatGLM6B.json)  

[evaluate数据：testata_ChatGLM6B.json](https://github.com/sheldonlll/FinetuneDataFiles/blob/main/testdata_ChatGLM6B.json)  
 
将ChatGLM-6B/ptuning/目录中的train.sh， evaluate.sh，web_demo.py和web_demo.sh文件替换  
切换到ChatGLM-6B/ptuning/目录中，执行命令：  

[1.bash ./train.sh](https://github.com/sheldonlll/FinetuneDataFiles/blob/main/train.sh)  

[2.bash ./web_demo.sh](https://github.com/sheldonlll/FinetuneDataFiles/blob/main/web_demo.sh) 


# LLAMA:  
1. 准备LLAMA模型：
   1.1 git lfs install
   1.2 git clone https://huggingface.co/huggyllama/llama-7b
    下载好的llama-7b存放在/root/autodl-tmp/v3/llama-7b/位置
2. 在data目录下新建MIT文件夹  
    2.1 在MIT文件夹下新建train_modify文件夹，并将finetune数据文件放入  
    2.2 在MIT文件夹下新建test文件夹，并将evaluate数据文件放入  

[finetune数据：traindata_LLAMA.json](https://github.com/sheldonlll/FinetuneDataFiles/blob/main/traindata_LMFlow.json)  

[evaluate数据：testdata_LLAMA.json](https://github.com/sheldonlll/FinetuneDataFiles/blob/main/testdata_LMFlow.json)  
 
3. 将scripts目录中的run_finetune.sh文件替换为run_finetune_LLAMA.sh  切换到scripts目录中，执行命令： 
修改的地方：
(
dataset_path=${project_dir}/data/MIT/train
--deepspeed examples/ds_config.json
)

[1. ./run_finetune_LLAMA.sh](https://github.com/sheldonlll/FinetuneDataFiles/blob/main/run_finetune.sh#L16)  

bug report:
'''
Traceback (most recent call last):
  File "/root/LMFlow/examples/finetune.py", line 65, in <module>
    main()
  File "/root/LMFlow/examples/finetune.py", line 61, in main
    tuned_model = finetuner.tune(model=model, dataset=dataset)
  File "/root/LMFlow/src/lmflow/pipeline/finetuner.py", line 285, in tune
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
  File "/root/miniconda3/envs/lmflow2/lib/python3.9/site-packages/transformers/trainer.py", line 1662, in train
    return inner_training_loop(
  File "/root/miniconda3/envs/lmflow2/lib/python3.9/site-packages/transformers/trainer.py", line 1731, in _inner_training_loop
    deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
  File "/root/miniconda3/envs/lmflow2/lib/python3.9/site-packages/transformers/deepspeed.py", line 378, in deepspeed_init
    deepspeed_engine, optimizer, _, lr_scheduler = deepspeed.initialize(**kwargs)
  File "/root/miniconda3/envs/lmflow2/lib/python3.9/site-packages/deepspeed/__init__.py", line 125, in initialize
    engine = DeepSpeedEngine(args=args,
  File "/root/miniconda3/envs/lmflow2/lib/python3.9/site-packages/deepspeed/runtime/engine.py", line 340, in __init__
    self._configure_optimizer(optimizer, model_parameters)
  File "/root/miniconda3/envs/lmflow2/lib/python3.9/site-packages/deepspeed/runtime/engine.py", line 1311, in _configure_optimizer
Loading extension module utils...
    self.optimizer = self._configure_bf16_optimizer(basic_optimizer)
  File "/root/miniconda3/envs/lmflow2/lib/python3.9/site-packages/deepspeed/runtime/engine.py", line 1501, in _configure_bf16_optimizer
    optimizer = BF16_Optimizer(
  File "/root/miniconda3/envs/lmflow2/lib/python3.9/site-packages/deepspeed/runtime/bf16_optimizer.py", line 92, in __init__
    self._setup_for_real_optimizer()
  File "/root/miniconda3/envs/lmflow2/lib/python3.9/site-packages/deepspeed/runtime/bf16_optimizer.py", line 112, in _setup_for_real_optimizer
    self._flatten_dense_tensors_aligned(
  File "/root/miniconda3/envs/lmflow2/lib/python3.9/site-packages/deepspeed/runtime/bf16_optimizer.py", line 258, in _flatten_dense_tensors_aligned
    return self.flatten(align_dense_tensors(tensor_list, alignment))
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 12.55 GiB (GPU 1; 23.69 GiB total capacity; 12.58 GiB already allocated; 10.38 GiB free; 12.58 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
Time to load utils op: 0.4046976566314697 seconds
Traceback (most recent call last):
  File "/root/LMFlow/examples/finetune.py", line 65, in <module>
    main()
  File "/root/LMFlow/examples/finetune.py", line 61, in main
    tuned_model = finetuner.tune(model=model, dataset=dataset)
  File "/root/LMFlow/src/lmflow/pipeline/finetuner.py", line 285, in tune
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
  File "/root/miniconda3/envs/lmflow2/lib/python3.9/site-packages/transformers/trainer.py", line 1662, in train
    return inner_training_loop(
  File "/root/miniconda3/envs/lmflow2/lib/python3.9/site-packages/transformers/trainer.py", line 1731, in _inner_training_loop
    deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
  File "/root/miniconda3/envs/lmflow2/lib/python3.9/site-packages/transformers/deepspeed.py", line 378, in deepspeed_init
    deepspeed_engine, optimizer, _, lr_scheduler = deepspeed.initialize(**kwargs)
  File "/root/miniconda3/envs/lmflow2/lib/python3.9/site-packages/deepspeed/__init__.py", line 125, in initialize
    engine = DeepSpeedEngine(args=args,
  File "/root/miniconda3/envs/lmflow2/lib/python3.9/site-packages/deepspeed/runtime/engine.py", line 340, in __init__
    self._configure_optimizer(optimizer, model_parameters)
  File "/root/miniconda3/envs/lmflow2/lib/python3.9/site-packages/deepspeed/runtime/engine.py", line 1311, in _configure_optimizer
    self.optimizer = self._configure_bf16_optimizer(basic_optimizer)
  File "/root/miniconda3/envs/lmflow2/lib/python3.9/site-packages/deepspeed/runtime/engine.py", line 1501, in _configure_bf16_optimizer
    optimizer = BF16_Optimizer(
  File "/root/miniconda3/envs/lmflow2/lib/python3.9/site-packages/deepspeed/runtime/bf16_optimizer.py", line 92, in __init__
    self._setup_for_real_optimizer()
  File "/root/miniconda3/envs/lmflow2/lib/python3.9/site-packages/deepspeed/runtime/bf16_optimizer.py", line 112, in _setup_for_real_optimizer
    self._flatten_dense_tensors_aligned(
  File "/root/miniconda3/envs/lmflow2/lib/python3.9/site-packages/deepspeed/runtime/bf16_optimizer.py", line 258, in _flatten_dense_tensors_aligned
    return self.flatten(align_dense_tensors(tensor_list, alignment))
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 12.55 GiB (GPU 0; 23.69 GiB total capacity; 12.58 GiB already allocated; 10.38 GiB free; 12.58 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
[2023-07-10 12:50:58,411] [INFO] [launch.py:318:sigkill_handler] Killing subprocess 1534
[2023-07-10 12:50:58,415] [INFO] [launch.py:318:sigkill_handler] Killing subprocess 1535
[2023-07-10 12:50:58,416] [ERROR] [launch.py:324:sigkill_handler] ['/root/miniconda3/envs/lmflow2/bin/python', '-u', '/root/LMFlow/examples/finetune.py', '--local_rank=1', '--model_name_or_path', '/root/autodl-tmp/v3/llama-7b/', '--dataset_path', '/root/LMFlow/data/MIT/train', '--output_dir', '/root/LMFlow/output_models/llama_finetune', '--overwrite_output_dir', '--num_train_epochs', '0.01', '--learning_rate', '2e-5', '--block_size', '128', '--per_device_train_batch_size', '1', '--deepspeed', 'examples/ds_config.json', '--bf16', '--run_name', 'finetune', '--validation_split_percentage', '0', '--logging_steps', '20', '--do_train', '--ddp_timeout', '72000', '--save_steps', '5000', '--dataloader_num_workers', '1'] exits with return code = 1
'''
