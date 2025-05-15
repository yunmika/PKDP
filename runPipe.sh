#! /bin/bash
# Author      :   Fuchuan Han
# Email       :   h2624366594@163.com
# Time        :   2025/02/10 20:39:40

# 检查是否有GPU和type参数
if [ $# -ne 1 ]; then
    # echo "runPipe.sh <device 0/1> <type signal/shuf> <subset 0.2/0.3/0.4/0.5/0.6/0.7/0.8/> <SNP 400/600/1200/2000/4000/6000/10000/20000/100000>"
    echo "runPipe.sh <device 0/1>"
    exit 1
fi

date

# 激活Pytorch_env环境
source /home/hanfc/anaconda3/bin/activate Pytorch_env

echo "Pytorch_env activated successfully."

# 运行脚本
subset=0.7
# type=$2
# 如果输入参数是0 则使用cuda:0，否则使用cuda:1
device=$1
# SNP=$4

if [ $device -eq 0 ]; then
    device=cuda:0
else
    device=cuda:1
fi
# SNP=10000
test_phe=/home/hanfc/workspace/Tour_GS_GWAS/3.GWAS/5.rMVP/rMVP_data_preparation/SampleCore_GCMS_2025/SampleCore_test_EN_${subset}.csv
train_phe=/home/hanfc/workspace/Tour_GS_GWAS/3.GWAS/5.rMVP/rMVP_data_preparation/SampleCore_GCMS_2025/SampleCore_train_EN_${subset}.csv
output_path=/home/hanfc/workspace/Tour_GS_GWAS/script2025/GS_script/DPlearning_space/PKDP/model_path/
prior_snp=/home/hanfc/workspace/Tour_GS_GWAS/3.GWAS/8.gemma/GCMS_Core_sample_2025/EN_${subset}
# geno_path=/home/hanfc/workspace/Tour_GS_GWAS/3.GWAS/8.gemma/GCMS_Core_sample_2025/EN_${subset}
# gmat_path=/home/hanfc/workspace/Tour_GS_GWAS/script2025/GS_script/DPlearning_space/PKDP/data/pop.kinship.csv
geno_path=/home/hanfc/workspace/Tour_GS_GWAS/1.variant_calling/11.hybri_filter/snp_impute/blocks_tagSNP/tag_2000_SNP.csv

# for i in {2..11}; do 
#     n=$((i + 1))  
#     phe=$(cut -f $n -d, "$test_phe" | head -n 1)  

#     echo "Running for phenotype $phe"
#     # Rscript ./msic/tk_sigSNP.R "$prior_snp/${phe}_Sig_u.snp" "$geno_path/${phe}_k/output/${phe}_${subset}_${type}_${SNP}.csv" "$geno_path/${phe}_k/output/Sig_SNP.csv"

#     ./PKDP.py --mode train \
#         --test_phe "$test_phe" \
#         --train_phe "$train_phe" \
#         --geno "$gmat_path" \
#         --pnum $i \
#         --output_path "$output_path" \
#         --seed 6 \
#         --epochs 200 \
#         --optuna_trials 50 \
#         --device $device \
#         --prefix "${phe}_${type}_${subset}_${SNP}_nocv" \
#         --early_stop \
#         --no_cv
# done

# for i in {2..11}; do 
#     n=$((i + 1))  
#     phe=$(cut -f $n -d, "$test_phe" | head -n 1)  

#     echo "Running for phenotype $phe"
#     # Rscript ./msic/tk_sigSNP.R "$prior_snp/${phe}_Sig_u.snp" "$gmat_path" "$geno_path/${phe}_k/output/Sig_SNP.csv"

#     ./PKDP.py --mode train \
#         --test_phe "$test_phe" \
#         --train_phe "$train_phe" \
#         --geno "$geno_path/${phe}_k/output/Sig_SNP.csv" \
#         --pnum $i \
#         --output_path "$output_path" \
#         --seed 6 \
#         --epochs 200 \
#         --optuna_trials 50 \
#         --device $device \
#         --prior_features_file "$prior_snp/${phe}_Sig.snp" \
#         --prefix "${phe}_${type}_${subset}_${SNP}_nocv" \
#         --early_stop \
#         --no_cv
# done

# date


# for i in LIM NER GER CIT CAR b.Pin a.Ter b.Phel; do 

#     echo "Running for phenotype $i"

#     ./PKDP.py --mode train \
#         --test_phe "$test_phe" \
#         --train_phe "$train_phe" \
#         --geno "$geno_path" \
#         --pnum $i \
#         --seed 6 \
#         --epochs 200 \
#         --optuna_trials 30 \
#         --batch_size 32 \
#         --device $device \
#         --output_path "$output_path" \
#         --prefix "${i}_nocv_opt" \
#         --early_stop \
#         --adjust_encoding \
#         --no_cv
# done

# date

for i in a.Ter; do 
# for i in NER GER CIT CAR; do 

    echo "Running for phenotype $i"

    # 检查是否存在$output_path/${i}_Sig_tempSNP.csv
    if [ ! -f "./Sig_path_TrSet/${i}_Sig_tempSNP.rr.csv" ]; then
        Rscript ./msic/tk_sigSNP.R "./Sig_path_TrSet/${i}_Sig.snp" "$geno_path" "./Sig_path_TrSet/${i}_Sig_tempSNP.rr.csv"
    fi

    ./PKDP.py --mode train \
        --test_phe "$test_phe" \
        --train_phe "$train_phe" \
        --geno "./Sig_path_TrSet/${i}_Sig_tempSNP.rr.csv" \
        --pnum $i \
        --seed 6 \
        --epochs 200 \
        --optuna_trials 30 \
        --batch_size 32 \
        --device $device \
        --output_path "$output_path" \
        --prefix "${i}_nocv_opt" \
        --early_stop \
        --adjust_encoding \
        --fc_units 128 64 32 \
        --main_channels 64 32 32 \
        --conv_kernel_size 11 7 5 \
        --no_cv
        # --prior_channels 16 32 32 \
        # --prior_kernel_size 7 7 7 \
        # --prior_features_file "./Sig_path_TrSet/${i}_Sig.snp.rr" \
done

date