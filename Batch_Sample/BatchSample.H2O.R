#!/dssg/home/weiyl/anaconda3/envs/H2O.ai/bin/Rscript

################################################################################################################################################
# 1. add xgboost and bayes, but bayes didn't return score, use it carefully;
# 2. 20210112: split data train v.s. test by c("Detail_Group", "Stage_TNM");
# 3. 20210115: add PCA50 to features;
# 4. 20210120: h2o.init(port = sample())
################################################################################################################################################

library(optparse)
option_list<-list(make_option(c("--FeatureType"), type="character", default=NULL, 
                              help="feature types, comma separated. ", metavar="character"),
                  make_option(c("--FeatureFile"), type="character", default=NULL, 
                              help="feature files, comma separated, each file is in CSV format, sample*feature. corresponds to --FeatureType.", 
                              metavar="character"),
                  make_option(c("--algorithms"), type="character", default="glm,gbm,rf,dl,xgboost", 
                              help="Algorithms used, comma separated. glm,gbm,rf,dl,xgboost", 
                              metavar="character"),
                  make_option(c("-o", "--output"), type="character", default="TMP",
                              help="results directory.", metavar="character"),
                  make_option(c("-l", "--list"), type="character", default=NULL, 
                              help="id info for feature file samples, table seperated.", 
                              metavar="character"),
                  make_option(c("--seed"), type="integer", default=1, 
                              help="random seed, default 1.", metavar="integer"),
                  make_option(c("--ratio"), type="double", default=0.6, 
                              help="Ratio to split data, 0~1. default 0.6", metavar="double"),
                  make_option(c("--addPCA"), type="logical", default=FALSE,
                              help="add top 50 PCs to each feature, PCA was calculated from train set, default FALSE.", metavar="logical"),
                  make_option(c("-n", "--nthreads"), type="integer", default=16, 
                              help="nthreads, default 16.", metavar="integer"),
                  make_option(c("-m", "--memory"), type="character", default="24G", 
                              help="memory used, default 24G.", metavar="character"),
                  make_option(c("--stack"), type="logical", default=TRUE,
                              help="stack all base models or not, default TRUE.", metavar="logical")
)

opt_parser<-OptionParser(option_list=option_list, description = "    Training models based on different features.")
opt<-parse_args(opt_parser)

library(h2o)
server_success = 0
while (server_success == 0){
    try({
        if (! is.null(opt$port)){
            h2o.init(nthreads = opt$nthreads, max_mem_size = opt$memory, port = opt$port)
        } else {
            h2o.init(nthreads = opt$nthreads, max_mem_size = opt$memory, port = sample(40000:65535,1))
        }
        server_success = server_success + 1
    })
}


tryCatch({
    library(dplyr)
    
    dir.create(opt$output, recursive = T)
    opt$output = normalizePath(opt$output)
    
    
    ##### id split, to make sure same sample in same set #####
    id.info.all = read.delim(opt$list, header = T, stringsAsFactors = F, check.names = F)
    colnames(id.info.all)[1:2] = c("Sample","Type")
    id.info = id.info.all[,c(1,2)]
    
    set.seed(opt$seed)
    sample_train = id.info %>% slice_sample(prop = opt$ratio)
    sample_test = subset(id.info, !(Sample %in% sample_train$Sample))
    
    ##### feature files #####
    feature_type_list = strsplit(opt$FeatureType, split = ",")[[1]]
    feature_type_list = gsub("_",".",feature_type_list)
    feature_files = strsplit(opt$FeatureFile, split = ",")[[1]]
    algorithms_type_list = strsplit(opt$algorithms, split = ",")[[1]]
    feature_data = list()
    for (i in 1:length(feature_type_list)){
        tmp_data = read.csv(feature_files[i],header = T)
        colnames(tmp_data) = paste0(feature_type_list[i], colnames(tmp_data))
        colnames(tmp_data)[1] = "Sample"
        tmp_data = inner_join(id.info, tmp_data, c("Sample"="Sample"))
        feature_data[[feature_type_list[i]]] = tmp_data
    }
    
    
    ##### add PCA to features #####
    train_list=list()
    test_list = list()
    for (feature_type in feature_type_list){
        tmp_fearue_df = feature_data[[feature_type]]
        rownames(tmp_fearue_df) = tmp_fearue_df$Sample
        
        if (opt$addPCA) {
            tmp_fearue_df = tmp_fearue_df %>% select(where(is.numeric))
            train_feature_df = tmp_fearue_df[rownames(tmp_fearue_df) %in% sample_train$Sample, ]
            
            pca_model = prcomp(t(na.omit(t(train_feature_df))))
            saveRDS(pca_model, file = paste0(opt$output, "/h2o_seed",opt$seed,"_", feature_type, "_train_PCA.rds"))
            
            train_df_pca = data.frame(predict(pca_model, newdata = train_feature_df)[,1:50])
            colnames(train_df_pca) = paste0(feature_type, colnames(train_df_pca))
            train_df_pca$Sample = rownames(train_df_pca)
            train_list[[feature_type]] = inner_join(feature_data[[feature_type]], train_df_pca, c("Sample"="Sample"))
            
            test_feature_df = tmp_fearue_df[rownames(tmp_fearue_df) %in% sample_test$Sample,]
            test_df_pca = data.frame(predict(pca_model, newdata = test_feature_df)[,1:50])
            colnames(test_df_pca) = paste0(feature_type, colnames(test_df_pca))
            test_df_pca$Sample = rownames(test_df_pca)
            test_list[[feature_type]] = inner_join(feature_data[[feature_type]], test_df_pca, c("Sample"="Sample")) 
            
        } else {
            train_list[[feature_type]] = tmp_fearue_df[rownames(tmp_fearue_df) %in% sample_train$Sample, ]
            test_list[[feature_type]] = tmp_fearue_df[rownames(tmp_fearue_df) %in% sample_test$Sample, ]
        }
    }
    
    combined_data_train = train_list[[1]]
    combined_data_test = test_list[[1]]
    if (length(train_list) > 1) {
        for (i in 2:length(train_list)){
            combined_data_train = inner_join(combined_data_train, 
                                             train_list[[i]][,setdiff(colnames(train_list[[i]]), "Type")], c("Sample" = "Sample"))
            combined_data_test = inner_join(combined_data_test, 
                                            test_list[[i]][,setdiff(colnames(test_list[[i]]), "Type")], c("Sample" = "Sample"))
        }
    }
    
    ##连续变量
    # combined_data_train$Type = as.factor(combined_data_train$Type)
    combined_data_train = as.h2o(combined_data_train)
    # combined_data_test$Type = as.factor(combined_data_test$Type)
    combined_data_test = as.h2o(combined_data_test)
    
    
    #### repeat base model train: 20 times #### 
    # initialize
    nfolds = 10
    model_list = list(stacked = list())
    performance_list = list(stacked = list())
    test_auc_df = data.frame(Run_test=character(), Feature=character(), model=character(), AUC=numeric(),stringsAsFactors=FALSE)
    for (feature_type in feature_type_list){
        model_list[[feature_type]] = list()
        performance_list[[feature_type]] = list()
    }
    
    train_score_df = as.data.frame(combined_data_train[,c("Sample","Type"),drop=F])
    colnames(train_score_df) = c("SampleID","var")
    
    test_score_df = as.data.frame(combined_data_test[,c("Sample","Type"),drop=F])
    colnames(test_score_df) = c("SampleID","var")
    
    for (feature_type in feature_type_list){
        #stratified split training and test dataset  
        tmp_train = train_list[[feature_type]]
        # tmp_train$Type = as.factor(tmp_train$Type)
        
        response <- "Type"
        predictor = setdiff(colnames(tmp_train), c("Sample","Type"))
        train_list[[feature_type]] <- as.h2o(tmp_train)

        
        #creating models
        print(paste0(feature_type, ": base models training ..."))
        if ("glm" %in% algorithms_type_list){
            print("    glm ...")
            model_list[[feature_type]][["glm"]] <- h2o.glm(x = predictor, y = response, training_frame = train_list[[feature_type]], nfolds = nfolds, 
                                                           seed = 1234, fold_assignment = "AUTO", keep_cross_validation_predictions = TRUE,
                                                           model_id = paste0(feature_type,"_","glm_seed",opt$seed))
        }
        if ("gbm" %in% algorithms_type_list){
            print("    gbm ...")
            model_list[[feature_type]][["gbm"]]<- h2o.gbm(x = predictor, y = response, training_frame = train_list[[feature_type]], nfolds = nfolds, 
                                                          seed = 1234, fold_assignment = "AUTO", keep_cross_validation_predictions = TRUE,
                                                          model_id = paste0(feature_type,"_","gbm_seed",opt$seed))
        }
        if("rf" %in% algorithms_type_list){
            print("    rf ...")
            model_list[[feature_type]][["rf"]] <- h2o.randomForest(x = predictor, y = response, training_frame = train_list[[feature_type]], 
                                                                   nfolds = nfolds,
                                                                   seed = 1234, fold_assignment = "AUTO", keep_cross_validation_predictions = TRUE,
                                                                   model_id = paste0(feature_type,"_","rf_seed",opt$seed))
        }
        if("dl" %in% algorithms_type_list){
            print("    dl ...")
            model_list[[feature_type]][["dl"]] <- h2o.deeplearning(x = predictor, y = response, training_frame = train_list[[feature_type]], 
                                                                   nfolds = nfolds, 
                                                                   seed = 1234, fold_assignment = "AUTO", keep_cross_validation_predictions = TRUE, 
                                                                   epochs = 50,
                                                                   model_id = paste0(feature_type,"_","dl_seed",opt$seed))
        }
        if("xgboost" %in% algorithms_type_list){
            print("    xgboost ...")
            model_list[[feature_type]][["xgboost"]] = h2o.xgboost(x = predictor, y = response, training_frame = train_list[[feature_type]], 
                                                                  nfolds = nfolds, 
                                                                  seed = 1234, fold_assignment = "AUTO", keep_cross_validation_predictions = TRUE,
                                                                  model_id = paste0(feature_type,"_","xgboost_seed",opt$seed))
        }

        if ("bayes" %in% algorithms_type_list){
            print("    bayes ...")
            model_list[[feature_type]][["bayes"]] = h2o.naiveBayes(x = predictor, y = response, training_frame = train_list[[feature_type]], 
                                                                   nfolds = nfolds,
                                                                   seed = 1234, fold_assignment = "AUTO", keep_cross_validation_predictions = TRUE,
                                                                   model_id = paste0(feature_type,"_","bayes_seed",opt$seed))
        }
        
        
        base_model_list = list()
        for (algorithms_type in algorithms_type_list){
            base_model_list = c(base_model_list, model_list[[feature_type]][[algorithms_type]])
        }
        
        model_list[[feature_type]][["stacked"]] <-  h2o.stackedEnsemble(x = predictor, y = response, 
                                                                        training_frame=train_list[[feature_type]],
                                                                        base_models = base_model_list, 
                                                                        seed=1234, 
                                                                        model_id = paste0(feature_type, "_stacked"),
                                                                        keep_levelone_frame=TRUE,
                                                                        metalearner_algorithm = "AUTO")
        
        if (nrow(combined_data_train) > 0){
          for (algorithms_type in names(model_list[[feature_type]])){
            #test predict score
            tmp_score_df = as.data.frame(h2o.predict(model_list[[feature_type]][[algorithms_type]], newdata = combined_data_train))
            colnames(tmp_score_df) = paste(feature_type, algorithms_type, colnames(tmp_score_df), sep = "_")
            train_score_df = cbind(train_score_df, tmp_score_df)
          }
        }
        
        if (nrow(combined_data_test) > 0){
            for (algorithms_type in names(model_list[[feature_type]])){
                #test predict score
                tmp_score_df = as.data.frame(h2o.predict(model_list[[feature_type]][[algorithms_type]], newdata = combined_data_test))
                colnames(tmp_score_df) = paste(feature_type, algorithms_type, colnames(tmp_score_df), sep = "_")
                test_score_df = cbind(test_score_df, tmp_score_df)
            }
        }
        
    }
    
    print("All featrues: base models done")
    
    #### base models stack #### 
    if (opt$stack & (length(feature_type_list) > 1)){
        base_model_list = list()
        
        for (feature_type in feature_type_list){
            for (algorithms_type in algorithms_type_list){
                base_model_list = c(base_model_list, model_list[[feature_type]][[algorithms_type]])
            }
        }
        
        print("All base models stacked ...")
        model_list[["stacked"]][["stacked"]] <- h2o.stackedEnsemble(y = response,
                                                                    training_frame=combined_data_train,
                                                                    base_models = base_model_list,
                                                                    seed=1234,
                                                                    model_id = "FeatureCombined_stacked",
                                                                    keep_levelone_frame=TRUE,
                                                                    metalearner_algorithm = "AUTO")
        
        #### evaluating stacked model performance ####
        if (nrow(combined_data_test) > 0){
            performance_list[["stacked"]][["stacked"]] <- h2o.performance(model_list[["stacked"]][["stacked"]], newdata = combined_data_test)
            
            tmp_confusionmatrix = as.data.frame(h2o.confusionMatrix(performance_list[["stacked"]][["stacked"]]))
            write.csv(tmp_confusionmatrix, file = paste0(opt$output, "/h2o_seed",opt$seed,"_","stacked","_","stacked","_ConfusionMarix.csv"),
                      row.names = T, quote = F)
            
            tmp_score_df = as.data.frame(h2o.predict(model_list[["stacked"]][["stacked"]], newdata = combined_data_test))
            colnames(tmp_score_df) = paste("stacked", "stacked", colnames(tmp_score_df), sep = "_")
            test_score_df = cbind(test_score_df, tmp_score_df)
        }
        
        print("Stack model done")
    }
    
    # save model_list and results, model was saved.
    save(list = c("model_list"), file=paste0(opt$output, "/h2o_seed",opt$seed,"_model_list.Rdata"))
    if (nrow(combined_data_train) > 0){
      write.csv(train_score_df, file=paste0(opt$output, "/h2o_seed",opt$seed,"_Train_Predict.csv"),row.names = F, quote = F)
    }
    if (nrow(combined_data_test) > 0){
        write.csv(test_score_df, file=paste0(opt$output, "/h2o_seed",opt$seed,"_Test_Predict.csv"),row.names = F, quote = F)
    }
    
    model_path = paste0(opt$output, "/h2o_seed",opt$seed,"_model_list.Rdata.dir")
    dir.create(model_path, recursive = T, showWarnings = F)
    stat = data.frame()
    for (feature_type in feature_type_list){
        stack_model = model_list[[feature_type]][["stacked"]]
        my_metrics = stack_model@model$training_metrics@metrics
        tmp.stat = data.frame("feature"=feature_type,
                            "MSE"=my_metrics[['MSE']],  #均方误差
                            "RMSE"=my_metrics[['RMSE']], #均方根误差(标准误差)
                            "MAE"=my_metrics[['mae']], #平均绝对误差
                            "R2"=my_metrics[['r2']],   #介于0~1,越接近1回归拟合越好。一般认为>0.8的模型拟合优度比较高
                            "mean_residual_deviance"=my_metrics[['mean_residual_deviance']] ,
                            'residual_deviance'=my_metrics[['residual_deviance']])
        stat = rbind(stat,tmp.stat)
        
        for (algorithms_type in names(model_list[[feature_type]])){
            h2o.saveModel(model_list[[feature_type]][[algorithms_type]], path = model_path, force = T)
        }
    }
    write.table(stat,file = paste0(opt$output, "/h2o_seed",opt$seed,"_model_metrics.tsv"),
                row.names = F,col.names = T,quote = F,sep = "\t")
    
    
    if (opt$stack & (length(feature_type_list) > 1)){
        h2o.saveModel(model_list[["stacked"]][["stacked"]], path = model_path, force = T)
    }
    
})

h2o.removeAll()
h2o.shutdown(prompt = FALSE)


