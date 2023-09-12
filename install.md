## How to install the dependencies

1. Create conda environment using:
	conda create -p /path/to/env/folder/env_name python=3.7
2. Activate the conda environment using:
	conda activate /path/to/env/folder/env_name
3. Install pytorch, torchvision etc
	conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c conda-forge
4. Install gxx_linux-64 using:
	conda install gxx_linux-64=9.3.0
5. Git clone inplace-abn and follow the steps in the repo to install. If it doesn't work then set CUDA_HOME
	export CUDA_HOME=/usr/local/cuda-11.3
6. If it doesn't work then try pip install inplace_abn
7. Make some changes to the setup.py to make inplace_abn compile for all family of GPUs.
   Do it only if the code throws error related to inplace-abn in your machine, otherwise ignore

   i) Add this line to the setup.py
   all_cuda_archs = cuda.get_gencode_flags().replace('compute=','arch=').split()
   ii) Replace this argument --> extra_compile_args={"cxx": ["-O3"], "nvcc": []},
       with --> extra_compile_args={"cxx": ["-O3"], "nvcc": all_cuda_archs},