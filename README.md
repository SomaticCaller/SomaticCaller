# User Guide of SomaticCaller

## Introduction

SomaticCaller is an innovative caller designed for the stable detection of variants.

## Install

The following packages and software need to be installed before running SomaticCaller.

[freebayes](https://github.com/freebayes/freebayes)

[lofreq](https://github.com/CSB5/lofreq)

[Mutect2](https://github.com/broadinstitute/gatk)

[sniper](https://gmt.genome.wustl.edu/packages/somatic-sniper/)

[strelka2](https://github.com/Illumina/strelka)

[VarDict](https://github.com/AstraZeneca-NGS/VarDictJava)

[VarScan](https://github.com/dkoboldt/varscan)

[h2o](https://github.com/h2oai/h2o-3)

## Usage

### Single Sample Mode

#### Training

```
Usage: SingleSample.H2O.R [options]
    Training models based on different features.

Options:
        --FeatureType=CHARACTER
                feature types, comma separated.

        --FeatureFile=CHARACTER
                feature files, comma separated, each file is in CSV format, sample*feature. corresponds to --FeatureType.

        --algorithms=CHARACTER
                Algorithms used, comma separated. glm,gbm,rf,dl,xgboost

        --sample_factors=CHARACTER
                balance classes ratio,comma separated

        -o CHARACTER, --output=CHARACTER
                results directory.

        -l CHARACTER, --list=CHARACTER
                id info for feature file samples, table seperated.

        --seed=INTEGER
                random seed, default 1.

        --ratio=DOUBLE
                Ratio to split data, 0~1. default 0.6

        --addPCA=LOGICAL
                add top 50 PCs to each feature, PCA was calculated from train set, default FALSE.

        -n INTEGER, --nthreads=INTEGER
                nthreads, default 16.

        -m CHARACTER, --memory=CHARACTER
                memory used, default 24G.

        --colnames=CHARACTER
                colomn names, comma separated.

        --stack=LOGICAL
                stack all base models or not, default TRUE.

        -h, --help
                Show this help message and exit
```

#### Prediction

```
Usage: SingleSample.predict.R [options]
    Predict new data based on models.

Options:
        --model=CHARACTER
                model, in Rdata format

        -o CHARACTER, --ResultDir=CHARACTER
                results directory.

        --NewFile=CHARACTER
                new data files, comma separated, each file is in CSV format, sample*feature. corresponds to --FeatureType. Default: NULL

        --FeatureType=CHARACTER
                feature types, comma separated.

        -l LIST, --list=LIST
                id info for samples

        --addPCA=LOGICAL
                add top 50 PCs to each feature, PCA was calculated from train set, default FALSE

        --datatype=CHARACTER
                data type.

        -n INTEGER, --nthreads=INTEGER
                nthreads, default 5.

        -m CHARACTER, --memory=CHARACTER
                memory used, default 12G.

        --stack=LOGICAL
                stack all base models or not, default TRUE.

        --seed=INTEGER
                random seed, default 1.

        -h, --help
                Show this help message and exit
```

### Batch Sample Mode

#### Training

```
Usage: BatchSample.H2O.R [options]
    Training models based on different features.

Options:
        --FeatureType=CHARACTER
                feature types, comma separated.

        --FeatureFile=CHARACTER
                feature files, comma separated, each file is in CSV format, sample*feature. corresponds to --FeatureType.

        --algorithms=CHARACTER
                Algorithms used, comma separated. glm,gbm,rf,dl,xgboost

        -o CHARACTER, --output=CHARACTER
                results directory.

        -l CHARACTER, --list=CHARACTER
                id info for feature file samples, table seperated.

        --seed=INTEGER
                random seed, default 1.

        --ratio=DOUBLE
                Ratio to split data, 0~1. default 0.6

        --addPCA=LOGICAL
                add top 50 PCs to each feature, PCA was calculated from train set, default FALSE.

        -n INTEGER, --nthreads=INTEGER
                nthreads, default 16.

        -m CHARACTER, --memory=CHARACTER
                memory used, default 24G.

        --stack=LOGICAL
                stack all base models or not, default TRUE.

        -h, --help
                Show this help message and exit
```

#### Prediction

```
Usage: BatchSample.predict.R [options]
    Predict new data based on models.

Options:
        --model=CHARACTER
                model, in Rdata format

        -o CHARACTER, --ResultDir=CHARACTER
                results directory.

        --NewFile=CHARACTER
                new data files, comma separated, each file is in CSV format, sample*feature. corresponds to --FeatureType. Default: NULL

        --FeatureType=CHARACTER
                feature types, comma separated.

        -l LIST, --list=LIST
                id info for samples

        --addPCA=LOGICAL
                add top 50 PCs to each feature, PCA was calculated from train set, default FALSE

        --datatype=CHARACTER
                data type.

        -n INTEGER, --nthreads=INTEGER
                nthreads, default 5.

        -m CHARACTER, --memory=CHARACTER
                memory used, default 12G.

        --case=CHARACTER
                case name when trainning model, default Cancer.

        --stack=LOGICAL
                stack all base models or not, default TRUE.

        --seed=INTEGER
                random seed, default 1.

        -h, --help
                Show this help message and exit
```

#### Optimization

```
usage: SAOpt.py [-h] -i INPUT [-o OUTPUT] [--init-temp [INIT_TEMP]] [--final-temp [FINAL_TEMP]] [--cooling-rate [COOLING_RATE]] [--rand-seed [RAND_SEED]]
          [--iterations [ITERATIONS]] [-g] [--greedy-init] [-v] [-vv]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input file (default: -i deviation.tsv)
  -o OUTPUT, --output OUTPUT
                        Output file (default: -o sparse.tsv)
  --init-temp [INIT_TEMP]
                        Initial temperature (default: --init-temp 1.0)
  --final-temp [FINAL_TEMP]
                        Final temperature (default: --final-temp 1e-7)
  --cooling-rate [COOLING_RATE]
                        Cooling rate (default: --cooling-rate 0.8)
  --rand-seed [RAND_SEED]
                        Random seed (default: --rand-seed 42)
  --iterations [ITERATIONS]
                        Number of iterations per temperature (default: --iterations 100)
  -g, --graphical       Display coverage graph.
  --greedy-init         Use greedy strategy for initial solution.
  -v, --verbose         Get more information.
  -vv, --debug          Get detailed debug information.
```

## Example

Here is an example of running SomaticCaller on a Linux system:

```bash
# Single Sample Mode:
# Training
Rscript SingleSample.H2O.R  --FeatureType=freebayes,lofreq,Mutect2,sniper,strelka2,VarDict,VarScan --FeatureFile=Feature/freebayes.csv,Feature/lofreq.csv,Feature/Mutect2.csv,Feature/sniper.csv,Feature/strelka2.csv,Feature/VarDict.csv,Feature/VarScan.csv --output=Results_TOO --list=InfoList/info.list --ratio=0.6 --seed=1 --nthreads=12 --memory=40G --addPCA=F --colnames=Detail_Group,Stage_TNM --sample_factors=1,1,1,1,1,1,1 
# Prediction
Rscript SingleSample.predict.R --model=Results_TOO/h2o_seed1_model_list.Rdata --ResultDir=Predict --list=InfoList/info.list --FeatureType=freebayes,lofreq,Mutect2,sniper,strelka2,VarDict,VarScan --NewFile=Feature/freebayes.csv,Feature/lofreq.csv,Feature/Mutect2.csv,Feature/sniper.csv,Feature/strelka2.csv,Feature/VarDict.csv,Feature/VarScan.csv --addPCA=F --datatype=all --seed=1


# Batch Sample Mode:
# Training
Rscript BatchSample.H2O.R --FeatureType=lofreq --FeatureFile=Feature/lofreq.csv --output=Results_freebayes_lofreq --list=InfoList/freebayes.info --ratio=0.6 --seed=7 --nthreads=10 --memory=30G --addPCA=F 
# Prediction
Rscript  BatchSample.predict.R --model=Results_freebayes_lofreq/h2o_seed7_model_list.Rdata --ResultDir=Predict --FeatureType=lofreq --NewFile=Feature/lofreq.csv  --addPCA=F --datatype=freebayes --seed=7 --list=info.list 
# Optimization
python SAOpt.py --input Predict/h2o_seed7_lofreq_Predict.csv --output Predict/h2o_seed7_lofreq_Predict_Opt.tsv --greedy-init -vv
```

## Output

The output of SomaticCaller is a VCF (Variant Call Format) file, conforming to the VCFv4.2 standard. This VCF file is generated by the recommended somatic variant caller, integrated with SomaticCaller. 

- CHROM: The chromosome number where the variant is located.

- POS: The position of the variant on the chromosome.

- ID: A unique identifier for the variant, if available.

- REF: The reference base(s) at the variant site.

- ALT: The alternate base(s) observed at the variant site.

- QUAL: Quality score of the variant call.

- FILTER: Filter status of the variant, indicating if it passes quality thresholds.

- INFO: Additional information about the variant, such as allele frequency, depth of coverage, and other annotations.

- FORMAT: Format of the data in the genotype fields.

- unknown: Sample-specific genotype information, detailing the genotype of the sample and additional metrics like genotype quality, depth, and allele count.
