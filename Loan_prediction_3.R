train=read.csv("train.csv")
test=read.csv("test.csv")
complete=rbind(train[,1:12],test)
complete$Loan_ID=NULL
complete$Loan_Amount_Term[complete$Loan_Amount_Term<360]=1
complete$Loan_Amount_Term[complete$Loan_Amount_Term>360]=0
complete$Loan_Amount_Term=as.factor(complete$Loan_Amount_Term)

#Imputation using missforest
complete$Credit_History=as.factor(complete$Credit_History)
imp_complete=missForest(complete,maxiter=10,ntree=500)
ncomplete=imp_complete$ximp
ntrain=ncomplete[1:614,]
ntest=ncomplete[615:981,]
ntrain$Loan_Status=train$Loan_Status

#feature selection using Boruta
install.packages("Boruta")
library(Boruta)
boruta.train=Boruta(Loan_Status~.,data=ntrain,doTrace=2)
boruta.train
final.boruta=TentativeRoughFix(boruta.train)
final.boruta
boruta.df=attStats(final.boruta)
boruta.df

#Recursive feature elimination
 library(randomForest)
library(caret)
 library(e1071)
library(plyr)
control=rfeControl(functions=rfFuncs,method="cv",number=10)
rfe.train=rfe(ntrain[,1:11],ntrain[,12],rfeControl=control)
rfe.train

# accuracy 0.7916 on test set 
model=randomForest(Loan_Status~Credit_History+ApplicantIncome+LoanAmount+CoapplicantIncome+Loan_Amount_Term+Property_Area,data=ntrain,ntree=1000)
# accuracy 0.7916 on test set
model=randomForest(Loan_Status~Credit_History+ApplicantIncome+LoanAmount+CoapplicantIncome+Loan_Amount_Term+Property_Area+Married,data=ntrain,ntree=1000)
# accuracy 0.7986
model=randomForest(Loan_Status~Credit_History+ApplicantIncome+LoanAmount+CoapplicantIncome+Loan_Amount_Term+Property_Area+Married+Dependents,data=ntrain,ntree=1000)

pred=predict(model,ntest)
pred=as.data.frame(pred)
pred$Loan_ID=test$Loan_ID
write.csv(pred,"sub3.csv",row.names=FALSE)

# Rank 59 out of 6606
