#!/dssg/home/weiyl/anaconda3/envs/H2O.ai/bin/Rscript

############################################################################################################################
# 1. 20210118, add --list, id.info.list;
# 2.           add --addPCA;
# 3.           fix bugs when sample number is ONE; pca predict add drop=F;
# 4. 20210630  restart when h2o failed;
# 5. 20210901  TOO models predict;
############################################################################################################################


library(optparse)
option_list<-list(make_option(c("--model"), type="character", default=NULL, 
                              help="model, in Rdata format", metavar="character"),
                  make_option(c("-o", "--ResultDir"), type="character", default="TMP",
                              help="results directory.", metavar="character"),
                  make_option(c("--NewFile"), type="character", default=NULL, 
                              help="new data files, comma separated, each file is in CSV format, sample*feature. corresponds to --FeatureType. Default: NULL", 
                              metavar="character"),
                  make_option(c("--FeatureType"), type="character", default=NULL, 
                              help="feature types, comma separated. ", metavar="character"),
                  make_option(c("-l", "--list"), type="character", default=NULL, 
                              help="id info for samples"),
                  make_option(c("--addPCA"), type="logical", default=FALSE,
                              help="add top 50 PCs to each feature, PCA was calculated from train set, default FALSE", metavar="logical"),
                  make_option(c("--datatype"), type="character", default="NewData", 
                              help="data type. ", metavar="character"),
                  make_option(c("-n", "--nthreads"), type="integer", default=5, 
                              help="nthreads, default 5.", metavar="integer"),
                  make_option(c("-m", "--memory"), type="character", default="12G", 
                              help="memory used, default 12G.", metavar="character"),
                  make_option(c("--stack"), type="logical", default=TRUE,
                              help="stack all base models or not, default TRUE.", metavar="logical"),
                  make_option(c("--seed"), type="integer", default=1, 
                              help="random seed, default 1.", metavar="integer")
)

opt_parser<-OptionParser(option_list=option_list, description = "    Predict new data based on models.")
opt<-parse_args(opt_parser)


library(h2o)
library(dplyr)
server_success = 0
while (server_success == 0){
    try({
        h2o.init(nthreads = opt$nthreads, max_mem_size = opt$memory, port = sample(40000:65535,1))
        server_success = server_success + 1
    })
}

try({
    dir.create(opt$ResultDir, recursive = T)
    
    feature_type_list = strsplit(opt$FeatureType, split = ",")[[1]]
    feature_type_list = gsub("_",".",feature_type_list)
    opt$model = normalizePath(opt$model)
    model_path = dirname(opt$model)
    load(opt$model)
    model_files = paste0(opt$model, ".dir/", list.files(paste0(opt$model, ".dir/")))
    tmp = sapply(model_files, h2o.loadModel)
    
    id.info.all = read.delim(opt$list, header = T, stringsAsFactors = F, check.names = F)
    colnames(id.info.all)[c(1,2)] = c("SampleID","Type")
    id.info = id.info.all[,c(1,2)]
    
    ##### new data set predict #####
    new_files = strsplit(opt$NewFile, split = ",")[[1]]
    new_data_list = list()
    for (i in 1:length(feature_type_list)){
        tmp_data = read.csv(new_files[i],header = T, stringsAsFactors = F)
        colnames(tmp_data) = paste0(feature_type_list[i], colnames(tmp_data))
        colnames(tmp_data)[1] = "SampleID"
        rownames(tmp_data) = tmp_data$SampleID
        tmp_data = tmp_data[rownames(tmp_data) %in% id.info$SampleID, ]
        
        if(opt$addPCA){
            pca_model_file = paste0(model_path, "/h2o_seed",opt$seed,"_", feature_type_list[i], "_train_PCA.rds")
            print(paste0("loading ", pca_model_file))
            pca_model = readRDS(pca_model_file)
            new_df_pca = data.frame(predict(pca_model, newdata = tmp_data)[,1:50,drop=F])
            colnames(new_df_pca) = paste0(feature_type_list[i], colnames(new_df_pca))
            new_df_pca$SampleID = rownames(new_df_pca)
            tmp_data = inner_join(tmp_data, new_df_pca, c("SampleID" = "SampleID"))
        } 
        
        new_data_list[[feature_type_list[i]]] = tmp_data
    }
    
    valid_df = new_data_list[[1]]
    if (length(new_data_list) > 1){
        for (i in 2:length(new_data_list)){
            valid_df = inner_join(valid_df, new_data_list[[i]], c("SampleID" = "SampleID"))
        }
    }
    
    valid_df = as.h2o(valid_df)
    valid_add_pred = as.data.frame(valid_df[,c("SampleID"),drop=F])
    
    ## predict based on all models
    for (feature_type in feature_type_list){
        for (algorithms_type in names(model_list[[feature_type]])){
            tmp_pred_df = as.data.frame(h2o.predict(model_list[[feature_type]][[algorithms_type]], newdata = valid_df))
            colnames(tmp_pred_df) = paste(feature_type, algorithms_type, colnames(tmp_pred_df),sep = "_" )
            valid_add_pred = cbind(valid_add_pred, tmp_pred_df)
        }
    }
    
    if (opt$stack & (length(feature_type_list) > 1)){
        tmp_pred_df = as.data.frame(h2o.predict(model_list[["stacked"]][["stacked"]], newdata = valid_df))
        colnames(tmp_pred_df) = paste("stacked", "stacked", colnames(tmp_pred_df), sep = "_")
        valid_add_pred = cbind(valid_add_pred, tmp_pred_df)
    }
    
    valid_add_pred =  inner_join(id.info, valid_add_pred, c("SampleID" = "SampleID"))
    write.csv(valid_add_pred, file=paste0(opt$ResultDir, "/h2o_seed",opt$seed,"_", opt$datatype, "_Predict.csv"),row.names = F, quote = F)
})

h2o.removeAll()
h2o.shutdown(prompt = FALSE)



