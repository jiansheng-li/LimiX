CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 inference_classifier.py --model_path /mnt/public/jianshengli/Limix/LimiX-16M.ckpt --inference_config_path config/cls_baseline_iclr.json --save_name baseline_tabarena
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 inference_classifier.py --model_path /mnt/public/jianshengli/Limix/LimiX-16M.ckpt --inference_config_path config/cls_cluster_iclr.json --save_name cluster_tabarena
CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 inference_classifier.py --model_path /mnt/public/jianshengli/Limix/LimiX-16M.ckpt --inference_config_path config/cls_cluster_iclr.json --save_name cluster_tabarena
torchrun --nproc_per_node=8 inference_classifier.py --model_path /mnt/public/jianshengli/Limix/LimiX-16M.ckpt --inference_config_path config/cls_default_iclr.json --save_name retrieval_tabarena





CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 iclr_inference.py  --inference_config_path config/cls_cluster_iclr.json --data_dir /mnt/public/classifier_benchmarks/bcco_106 --use_threshold --use_cluster
CUDA_VISIBLE_DEVICES=4 torchrun --master_port 29514 --nproc_per_node=1 iclr_inference.py  --inference_config_path config/cls_cluster_iclr.json --data_dir /mnt/public/classifier_benchmarks/tabzilla --use_threshold --use_cluster
CUDA_VISIBLE_DEVICES=5 torchrun --master_port 29515 --nproc_per_node=1 iclr_inference.py  --inference_config_path config/cls_cluster_iclr.json --data_dir /mnt/public/classifier_benchmarks/talent_addition_classifier --use_threshold --use_cluster

CUDA_VISIBLE_DEVICES=6 torchrun --master_port 29516 iclr_inference_reg.py  --inference_config_path config/reg_cluster_iclr.json --data_dir /mnt/public/regression_benchmarks/benchmark_regression --use_threshold --use_cluster
CUDA_VISIBLE_DEVICES=7 torchrun --master_port 29517 --nproc_per_node=1 iclr_inference_reg.py  --inference_config_path config/reg_cluster_iclr.json --data_dir /mnt/public/regression_benchmarks/ctr23 --use_threshold --use_cluster
CUDA_VISIBLE_DEVICES=4 torchrun --master_port 29514 --nproc_per_node=1 iclr_inference_reg.py  --inference_config_path config/reg_cluster_iclr.json --data_dir /mnt/public/regression_benchmarks/talent_regression --use_threshold --use_cluster






CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 iclr_inference.py  --inference_config_path config/cls_cluster_iclr.json --data_dir /mnt/public/classifier_benchmarks/bcco_106 --use_threshold --use_cluster
CUDA_VISIBLE_DEVICES=1 torchrun --master_port 29511 --nproc_per_node=1 iclr_inference.py  --inference_config_path config/cls_cluster_iclr.json --data_dir /mnt/public/classifier_benchmarks/tabarena_classification --use_threshold --use_cluster
CUDA_VISIBLE_DEVICES=2 torchrun --master_port 29512 --nproc_per_node=1 iclr_inference.py  --inference_config_path config/cls_cluster_iclr.json --data_dir /mnt/public/classifier_benchmarks/tabzilla --use_threshold --use_cluster
CUDA_VISIBLE_DEVICES=3 torchrun --master_port 29513 --nproc_per_node=1 iclr_inference.py  --inference_config_path config/cls_cluster_iclr.json --data_dir /mnt/public/classifier_benchmarks/talent_addition_classifier --use_threshold --use_cluster


CUDA_VISIBLE_DEVICES=4 torchrun --master_port 29514 --nproc_per_node=1 iclr_inference.py  --inference_config_path config/cls_cluster_iclr.json --data_dir /mnt/public/classifier_benchmarks/bcco_106 --use_cluster
CUDA_VISIBLE_DEVICES=5 torchrun --master_port 29515 --nproc_per_node=1 iclr_inference.py  --inference_config_path config/cls_cluster_iclr.json --data_dir /mnt/public/classifier_benchmarks/tabarena_classification --use_cluster
CUDA_VISIBLE_DEVICES=6 torchrun --master_port 29516 --nproc_per_node=1 iclr_inference.py  --inference_config_path config/cls_cluster_iclr.json --data_dir /mnt/public/classifier_benchmarks/tabzilla --use_cluster
CUDA_VISIBLE_DEVICES=7 torchrun --master_port 29517 --nproc_per_node=1 iclr_inference.py  --inference_config_path config/cls_cluster_iclr.json --data_dir /mnt/public/classifier_benchmarks/talent_addition_classifier --use_cluster


CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 iclr_inference_reg.py  --inference_config_path config/reg_cluster_iclr.json --data_dir /mnt/public/regression_benchmarks/benchmark_regression --use_threshold --use_cluster
CUDA_VISIBLE_DEVICES=2 torchrun --master_port 29511 --nproc_per_node=1 iclr_inference_reg.py  --inference_config_path config/reg_cluster_iclr.json --data_dir /mnt/public/regression_benchmarks/ctr23 --use_threshold --use_cluster
CUDA_VISIBLE_DEVICES=3 torchrun --master_port 29512 --nproc_per_node=1 iclr_inference_reg.py  --inference_config_path config/reg_cluster_iclr.json --data_dir /mnt/public/regression_benchmarks/talent_regression --use_threshold --use_cluster

CUDA_VISIBLE_DEVICES=4 torchrun --master_port 29513 --nproc_per_node=1 iclr_inference_reg.py  --inference_config_path config/reg_cluster_iclr.json --data_dir /mnt/public/regression_benchmarks/benchmark_regression --use_cluster
CUDA_VISIBLE_DEVICES=5 torchrun --master_port 29514 --nproc_per_node=1 iclr_inference_reg.py  --inference_config_path config/reg_cluster_iclr.json --data_dir /mnt/public/regression_benchmarks/ctr23 --use_cluster
CUDA_VISIBLE_DEVICES=6 torchrun --master_port 29515 --nproc_per_node=1 iclr_inference_reg.py  --inference_config_path config/reg_cluster_iclr.json --data_dir /mnt/public/regression_benchmarks/talent_regression --use_cluster



CUDA_VISIBLE_DEVICES=6 torchrun --master_port 29516 inference_classifier.py  --inference_config_path config/cls_cluster_iclr.json --use_threshold --use_cluster



chicago-food-inspections
covid19-dataset
boxing-matches
airlines-dataset-to-predict-a-delay
cardio-vascular-disease-detection
employee-attrition-dataset
kickstarter-projects
loantap-data
roadside-noise-level-dataset-with-labels
stellar-classification-dataset-sdss17
credit-Card-fraud
titanic-huge-dataset-1m-passengers
expedia-travel-dataset
smoking-drinking-dataset




