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


# LLAMA:  
1. 准备LLAMA模型：
- git lfs install
- git clone https://huggingface.co/huggyllama/llama-7b
- 下载好的llama-7b存放在/root/autodl-tmp/v3/llama-7b/位置
2. 在data目录下新建MIT文件夹  
- 在MIT文件夹下新建train_modify文件夹，并将finetune数据文件放入  
- 在MIT文件夹下新建test文件夹，并将evaluate数据文件放入  
- [finetune数据：traindata_LLAMA.json](https://github.com/sheldonlll/FinetuneDataFiles/blob/main/traindata_LMFlow.json)  
- [evaluate数据：testdata_LLAMA.json](https://github.com/sheldonlll/FinetuneDataFiles/blob/main/testdata_LMFlow.json)  
 
3. 将scripts目录中的run_finetune.sh文件替换为run_finetune_LLAMA.sh  切换到scripts目录中，执行命令： 

- [./run_finetune_LLAMA.sh](https://github.com/sheldonlll/FinetuneDataFiles/blob/main/run_finetune.sh#L16)  

bug report: https://chat.openai.com/share/f5e953a9-b335-4c8f-8c75-13bd8250346d  

' 
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 12.55 GiB (GPU 1; 23.69 GiB total capacity; 12.58 GiB already allocated; 10.38 GiB free; 12.58 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
Time to load utils op: 0.4046976566314697 seconds

torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 12.55 GiB (GPU 0; 23.69 GiB total capacity; 12.58 GiB already allocated; 10.38 GiB free; 12.58 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF  

[2023-07-10 12:50:58,411] [INFO] [launch.py:318:sigkill_handler] Killing subprocess 1534
[2023-07-10 12:50:58,415] [INFO] [launch.py:318:sigkill_handler] Killing subprocess 1535
[2023-07-10 12:50:58,416] [ERROR] [launch.py:324:sigkill_handler] ['/root/miniconda3/envs/lmflow2/bin/python', '-u', '/root/LMFlow/examples/finetune.py', '--local_rank=1', '--model_name_or_path', '/root/autodl-tmp/v3/llama-7b/', '--dataset_path', '/root/LMFlow/data/MIT/train', '--output_dir', '/root/LMFlow/output_models/llama_finetune', '--overwrite_output_dir', '--num_train_epochs', '0.01', '--learning_rate', '2e-5', '--block_size', '128', '--per_device_train_batch_size', '1', '--deepspeed', 'examples/ds_config.json', '--bf16', '--run_name', 'finetune', '--validation_split_percentage', '0', '--logging_steps', '20', '--do_train', '--ddp_timeout', '72000', '--save_steps', '5000', '--dataloader_num_workers', '1'] exits with return code = 1
'


- run_finetune修改的地方：

```
dataset_path=${project_dir}/data/MIT/train

--deepspeed examples/ds_config.json
```


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
