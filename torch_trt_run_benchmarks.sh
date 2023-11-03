#!/bin/bash

pip install torchaudio numba
cd /opt/TorchBench
python install.py dlrm llama_7b detectron2_fasterrcnn_r_50_fpn

large_model_batch_sizes=(1 2 4 8 16 32 64)

# Benchmark DLRM model Eager
echo "Benchmarking DLRM model Eager"
for bs in ${large_model_batch_sizes[@]}
do
  python run.py dlrm --precision fp16 -d cuda -t eval --torchdynamo eager --bs ${bs}
done
echo "Completed Benchmarking DLRM model Eager"

# Benchmark DLRM model Inductor
echo "Benchmarking DLRM model Inductor"
for bs in ${large_model_batch_sizes[@]}
do
  python run.py dlrm --precision fp16 -d cuda -t eval --torchdynamo inductor --torchinductor_enable_max_autotune_gemm --bs ${bs}
echo "Completed Benchmarking DLRM model Inductor"
done

# Benchmark DLRM model Torch-TRT
echo "Benchmarking DLRM model Torch-TRT"
for bs in ${large_model_batch_sizes[@]}
do
  python run.py dlrm --precision fp16 -d cuda -t eval --backend torch_trt --bs 1 --truncate_long_and_double --ir torch_compile
done
echo "Benchmarking Completed DLRM model Torch-TRT"


# Benchmark Llama-7B model Eager
echo "Benchmarking Llama-7B model Eager"
for bs in ${large_model_batch_sizes[@]}
do
  python run.py llama_v2_7b --precision fp16 -d cuda -t eval --torchdynamo eager --bs ${bs}
done
echo "Benchmarking Completed Llama-7B model Eager"

# Benchmark Llama-7B model Inductor
echo "Benchmarking Llama-7B model Inductor"
for bs in ${large_model_batch_sizes[@]}
do
  python run.py llama_v2_7b --precision fp16 -d cuda -t eval --torchdynamo inductor --torchinductor_enable_max_autotune_gemm --bs ${bs}
echo "Benchmarking Completed Llama-7B model Inductor"
done

# Benchmark Llama-7B model Torch-TRT
echo "Benchmarking Llama-7B model Torch-TRT"
for bs in ${large_model_batch_sizes[@]}
do
  python run.py llama_v2_7b --precision fp16 -d cuda -t eval --backend torch_trt --bs 1 --truncate_long_and_double --ir torch_compile
echo "Benchmarking Llama-Completed 7B model Torch-TRT"
done


# Benchmark Detectron2-50-FPN model Eager
echo "Benchmarking Detectron2-50-FPN model Eager"
for bs in ${large_model_batch_sizes[@]}
do
  python run.py detectron2_fasterrcnn_r_50_fpn --precision fp16 -d cuda -t eval --torchdynamo eager --bs ${bs}
done
echo "Benchmarking Detectron2-Completed 50-FPN model Eager"

# Benchmark Detectron2-50-FPN model Inductor
echo "Benchmarking Detectron2-50-FPN model Inductor"
for bs in ${large_model_batch_sizes[@]}
do
  python run.py detectron2_fasterrcnn_r_50_fpn --precision fp16 -d cuda -t eval --torchdynamo inductor --torchinductor_enable_max_autotune_gemm --bs ${bs}
echo "Benchmarking Detectron2-Completed 50-FPN model Inductor"
done

# Benchmark Detectron2-50-FPN model Torch-TRT
echo "Benchmarking Detectron2-50-FPN model Torch-TRT"
for bs in ${large_model_batch_sizes[@]}
do
  python run.py detectron2_fasterrcnn_r_50_fpn --precision fp16 -d cuda -t eval --backend torch_trt --bs 1 --truncate_long_and_double --ir torch_compile
echo "Benchmarking Detectron2-50-Completed FPN model Torch-TRT"
done
