##Libraries
pkgTest <- function(pkg){
  new.pkg <- pkg[!(pkg %in% installed.packages()[,  "Package"])]
  if (length(new.pkg)) 
    install.packages(new.pkg,  dependencies = TRUE)
  sapply(pkg,  require, warn.conflicts=F, quietly=T, character.only = TRUE)
}

if(!require(devtools)) install.packages("devtools")
library(devtools)
install_github("mroberts/stmBrowser",dependencies=TRUE)

lapply(c("tidyverse", "quanteda", "quanteda", "quanteda.textstats",
         "quanteda.textplots", "readtext", "stringi", "textstem", "lubridate",
         "caret", "MLmetrics", "doParallel", "naivebayes", "stm", "wordcloud",
         "stmBrowser", "LDAvis"), pkgTest)

#Read Data
data <- read.csv("./data/yelp_data_small.csv", 
                 stringsAsFactors=FALSE,
                 encoding = "utf-8")

?require
#Question 1.1

#Replace the return character "\n" with a whitespace
data$text <- gsub("\\\\n", " ", data$text)
#Replace the resulting double whitespaces with singles
data$text <- gsub("  ", " ", data$text)
#check for duplicates (there are none)
which(duplicated(data$text))
#no natural ID field supplied so just specify the text column
corp <- corpus(data, text_field = 'text')

#Create a summary object to view the corpus attributes
corpSum <- summary(corp, n = nrow(docvars(corp)))
head(corpSum)

#Print proportions of positive and negative tokens vs reviews
table(data$sentiment)/10000
paste("Proportion of Positive Tokens:",
      sum(corpSum[corpSum$sentiment=='pos',]$Tokens)/sum(corpSum$Tokens))
paste("Proportion of Negative Tokens:",
      sum(corpSum[corpSum$sentiment=='neg',]$Tokens)/sum(corpSum$Tokens))


#Question 1.2

#text processing
toks <- quanteda::tokens(corp, 
                         include_docvars = TRUE,
                         remove_numbers = TRUE,
                         remove_punct = TRUE,
                         remove_symbols = TRUE,
                         remove_separators = TRUE,
                         remove_url = TRUE) %>%
  tokens_tolower() %>%
  tokens_remove(stopwords("english"))

#collocations
clcs <- textstat_collocations(toks, size = 3, min_count = 20)['z' > 3.09,]
toks <- tokens_compound(toks, clcs)
clcs <- textstat_collocations(toks, size = 2, min_count = 20)['z' > 3.09,]
toks <- tokens_compound(toks, clcs)
#Remove whitespaces
toks <- tokens_remove(quanteda::tokens(toks), "") 
#Lemmatise the tokens
toks <- as.tokens(lapply(as.list(toks), lemmatize_words))

#Create a preliminary dfm
dfmTest <- dfm(toks)
#Use preliminary dfm to find other stopwords and recycle code to create the final dfm
topfeatures(dfmTest, n=50)

new_stopwords <- (c('get','just','say','even','tell','us','can','call','use'))
toks <- tokens_remove(quanteda::tokens(toks), new_stopwords)
#create main dfm and add sentiment
dfm <- dfm(toks)
dfm$sentiment <- data$sentiment
#trim the dfm to decrease processing time
dfm <- dfm_trim(dfm, min_docfreq = 40)
#weight the dfm using tf-idf to improve predictive power
dfm <- dfm_tfidf(dfm)


#Question 1.3

#Convert the dfm into a data-frame to be used for machine learning
mlDat <- convert(dfm, to = "data.frame", docvars = NULL)[,-1]
#Label the sentiment to add to data frame
sntmntLabels <- dfm@docvars$sentiment
mlDat <- as.data.frame(cbind(sntmntLabels, mlDat))
#set seed for replicability
set.seed(2023)
#randomly reorder the documents
mlDat <- mlDat[sample(nrow(mlDat)), ]
#bound the validation set of 5%
valDat <- mlDat[1:round(nrow(mlDat) * 0.05),]
#the labelled data less the validation set
mainDat <- mlDat[(nrow(valDat)+1):nrow(mlDat),]

# create the specified split
partition <- createDataPartition(mainDat$sntmntLabels, p=0.8, list=FALSE)
Train <- mainDat[partition, ]
Test <- mainDat[-partition, ]
# create a control for the training data
ctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 3,
  classProbs= TRUE, summaryFunction = multiClassSummary,
  selectionFunction = "best", verboseIter = TRUE)

#Question 1.4

# create a custom tuning grid to avoid overfitting (???)
tuneGrid <- expand.grid(laplace = c(0,0.5,1.0),
                        usekernel = c(TRUE, FALSE),
                        adjust=c(0.75, 1, 1.25, 1.5))

#Assign 4 cores to the process (my i3 CPU is only quad-core)
cl <- makePSOCKcluster(4)
registerDoParallel(cl)
#Train the Naive Bayes model
firstTrain <- train(sntmntLabels ~ ., 
                  data = Train,  
                  method = "naive_bayes", 
                  metric = "F1",
                  trControl = ctrl,
                  tuneGrid = tuneGrid,
                  allowParallel= TRUE
)
saveRDS(firstTrain, "data/firstTrain")
#stop multiple R sessions
stopCluster(cl)

print(firstTrain)
# generate prediction on Test set using training set model
prediction <- predict(firstTrain, newdata = Test)
head(prediction) # first few predictions
head(Test)
confusionMatrix(reference = as.factor(Test$sntmntLabels),
                data = prediction, mode='everything')

finalTrain <- train(sntmntLabels ~ ., 
                  data = mainDat,  
                  method = "naive_bayes", 
                  trControl = trainControl(method = "none"),
                  tuneGrid = data.frame(firstTrain$bestTune))

print(finalTrain)
saveRDS(finalTrain, "data/finalTrain")

prediction2 <- predict(finalTrain, newdata = valDat)
head(prediction2) # first few predictions
head(valDat$sntmntLabels)
confusionMatrix(reference = as.factor(valDat$sntmntLabels),
                data = prediction2,
                mode='everything')





#Question 2.1
setwd(getwd())
bbData <- read.csv("./data/breitbart_2016_sample.csv", 
                 stringsAsFactors=FALSE,
                 encoding = "utf-8")


bbData <- bbData %>%
  mutate(date = dmy(date))
#Remove all of Breitbart's LiveWire blogs as the formatting renders they are mostly embedded Tweets
bbData <- bbData[-grep('LiveWire', bbData$title),]
#Remove URLs from the text including Twitter embedded pictures
bbData$content <- gsub("((https?://|www\\.)\\S+|pic\\.twitter\\.com/\\S+)",
                     "", bbData$content)

bbCorp <- corpus(bbData, text_field = 'content', docid_field = 'title')



bbToks <- quanteda::tokens(bbCorp, 
                         include_docvars = TRUE,
                         remove_numbers = TRUE,
                         remove_punct = TRUE,
                         remove_symbols = TRUE,
                         remove_separators = TRUE,
                         remove_url = TRUE) %>%
  tokens_tolower() %>%
  tokens_remove(stopwords("english"))

#collocations
bbClcs <- textstat_collocations(bbToks, size = 3, min_count = 20)['z' > 3.09,]
toks <- tokens_compound(bbToks, bbClcs)
bbClcs <- textstat_collocations(bbToks, size = 2, min_count = 20)['z' > 3.09,]
toks <- tokens_compound(bbToks, bbClcs)
#Remove whitespaces
bbToks <- tokens_remove(quanteda::tokens(bbToks), "") 
#Lemmatise the tokens
bbToks <- as.tokens(lapply(as.list(bbToks), lemmatize_words))

#Create a preliminary dfm
bbDfmTest <- dfm(bbToks)
#Use preliminary dfm to find other stopwords and recycle code to create the final dfm
topfeatures(bbDfmTest, n=50)

bb_stopwords <- (c('also','just','year','one','tell','two','can','call','get'))
bbToks <- tokens_remove(quanteda::tokens(bbToks), bb_stopwords)
#create main dfm and add dates
bbDfm <- dfm(bbToks)
bbDfm$date <- bbData$date

#trim the dfm in line with the question requirments
bbDfm <- dfm_trim(bbDfm, min_docfreq = 20)

#Question 2.2

# create STM
stmdfm <- convert(bbDfm, to = "stm")
K <- 35
bbModel <- stm(documents = stmdfm$documents,
                vocab = stmdfm$vocab,
                K = K,
                prevalence = ~ month(date),
                #prevalence = ~ source + s(as.numeric(date_month)), 
                data = stmdfm$meta,
                max.em.its = 500,
                init.type = "Spectral",
                seed = 2023,
                verbose = TRUE)

# Save your model!
saveRDS(bbModel, "data/bbModel")

labelTopics(bbModel)

cloud(bbModel,
      topic = 2,
      scale = c(2.5, 0.3),
      max.words = 50)
