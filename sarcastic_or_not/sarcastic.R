train=read.csv("train.csv",stringsAsFactors=FALSE)
test=read.csv("test.csv",stringsAsFactors=FALSE)
library(text2vec)
library(data.table)


prep_fun = tolower
tok_fun = word_tokenizer

it_train = itoken(train$tweet,preprocessor = prep_fun,tokenizer = tok_fun,ids = train$ID,progressbar = FALSE)
vocab = create_vocabulary(it_train)
vectorizer = vocab_vectorizer(vocab)
t1 = Sys.time()
dtm_train = create_dtm(it_train, vectorizer)
print(difftime(Sys.time(), t1, units = 'sec'))
#429.2 sec
dim(dtm_train)
# 91298 39855
library(glmnet)
t1 = Sys.time()
glmnet_classifier = cv.glmnet(x = dtm_train, y = train$label,family = 'binomial',alpha = 1,type.measure = "auc",nfolds =4 ,thresh = 1e-3, maxit = 1e3)
print(difftime(Sys.time(), t1, units = 'sec'))
#358.7
plot(glmnet_classifier)
print(paste("max AUC =", round(max(glmnet_classifier$cvm), 4)))
#.9506

# AFter pruning
pruned_vocab = prune_vocabulary(vocab,term_count_min = 10,doc_proportion_max = 0.05,doc_proportion_min = 0.001)
vectorizer = vocab_vectorizer(pruned_vocab)
t1 = Sys.time()
dtm_train  = create_dtm(it_train, vectorizer)
print(difftime(Sys.time(), t1, units = 'sec'))
#59.58
dim(dtm_train)
# 91298 1053
t1 = Sys.time()
glmnet_classifier = cv.glmnet(x = dtm_train, y = train$label,family = 'binomial',alpha = 1,type.measure = "auc",nfolds =4 ,thresh = 1e-3, maxit = 1e3)
print(difftime(Sys.time(), t1, units = 'sec'))
#311.6251
plot(glmnet_classifier)
print(paste("max AUC =", round(max(glmnet_classifier$cvm), 4)))
#.9006


#N Grams
vocab = create_vocabulary(it_train, ngram = c(1L, 2L))
vocab = vocab %>% prune_vocabulary(term_count_min = 10,doc_proportion_max = 0.5)
bigram_vectorizer = vocab_vectorizer(vocab)
dtm_train = create_dtm(it_train, bigram_vectorizer)
dim(dtm_train)
#91298 18544
t1 = Sys.time()
glmnet_classifier = cv.glmnet(x = dtm_train, y = train$label,family = 'binomial',alpha = 1,type.measure = "auc",nfolds =4 ,thresh = 1e-3, maxit = 1e3)
print(difftime(Sys.time(), t1, units = 'sec'))
#114.8395
plot(glmnet_classifier)
print(paste("max AUC =", round(max(glmnet_classifier$cvm), 4)))
#0.9635


#feature hashing
h_vectorizer = hash_vectorizer(hash_size = 2 ^ 14, ngram = c(1L, 2L))
dtm_train = create_dtm(it_train, h_vectorizer)
dim(dtm_train)
#91298 16384
t1 = Sys.time()
glmnet_classifier = cv.glmnet(x = dtm_train, y = train$label,family = 'binomial',alpha = 1,type.measure = "auc",nfolds =4 ,thresh = 1e-3, maxit = 1e3)
print(difftime(Sys.time(), t1, units = 'sec'))
#119.0621
plot(glmnet_classifier)
print(paste("max AUC =", round(max(glmnet_classifier$cvm), 4)))
#0.9466


# Applying NGrams on test data as it gave best auc on train data and it is regularised
it_test = test$tweet %>% prep_fun %>% tok_fun %>% itoken(ids = test$ID,progressbar = FALSE)
dtm_test = create_dtm(it_test, bigram_vectorizer)
preds = predict(glmnet_classifier, dtm_test, type = 'response')[,1]

# leaderboard rank 31



