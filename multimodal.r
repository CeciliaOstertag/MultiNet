

library("matrixStats")

auc <- function(y){
  n <- length(y)
  0.5*(1/58)*(y[1]+y[n]+2*sum(y[-c(1,n)]))
}
########## PLOT TRAINING CURVES

dataList=list("clinical","multimodal","multimodal_adapt","clinical_manq","multimodal_manq","multimodal_adapt_manq")
for(name in dataList)
{
  log_1 = read.table(paste("~/Bureau/RScripts/all/",name,"_0.csv",sep=""), header=TRUE, sep = ",")
  log_1 = cbind(c(1:75),log_1)
  colnames(log_1)[1] = "epoch"
  log_2 = read.table(paste("~/Bureau/RScripts/all/",name,"_1.csv",sep=""), header=TRUE, sep = ",")
  log_2 = cbind(c(1:75),log_2)
  colnames(log_2)[1] = "epoch"
  log_3 = read.table(paste("~/Bureau/RScripts/all/",name,"_2.csv",sep=""), header=TRUE, sep = ",")
  log_3 = cbind(c(1:75),log_3)
  colnames(log_3)[1] = "epoch"
  log_4 = read.table(paste("~/Bureau/RScripts/all/",name,"_3.csv",sep=""), header=TRUE, sep = ",")
  log_4 = cbind(c(1:75),log_4)
  colnames(log_4)[1] = "epoch"
  
  loss = as.data.frame(colMeans(rbind(log_1$loss,log_2$loss,log_3$loss,log_4$loss)))
  colnames(loss)[1] = "loss"
  loss_std = as.data.frame(colSds(rbind(log_1$loss,log_2$loss,log_3$loss,log_4$loss)))
  colnames(loss_std)[1] = "loss_std"
  valloss = as.data.frame(colMeans(rbind(log_1$valloss,log_2$valloss,log_3$valloss,log_4$valloss)))
  colnames(valloss)[1] = "valloss"
  valloss_std = as.data.frame(colSds(rbind(log_1$valloss,log_2$valloss,log_3$valloss,log_4$valloss)))
  colnames(valloss_std)[1] = "valloss_std"
  print(valloss[75,1])
  print(valloss_std[75,1])
  acc = as.data.frame(colMeans(rbind(log_1$acc,log_2$acc,log_3$acc,log_4$acc)))
  colnames(acc)[1] = "acc"
  acc_std = as.data.frame(colSds(rbind(log_1$acc,log_2$acc,log_3$acc,log_4$acc)))
  colnames(acc_std)[1] = "acc_std"
  valacc = as.data.frame(colMeans(rbind(log_1$valacc,log_2$valacc,log_3$valacc,log_4$valacc)))
  colnames(valacc)[1] = "valacc"
  valacc_std = as.data.frame(colSds(rbind(log_1$valacc,log_2$valacc,log_3$valacc,log_4$valacc)))
  colnames(valacc_std)[1] = "valacc_std"
  print(valacc[75,1])
  print(valacc_std[75,1])
  f1 = as.data.frame(colMeans(rbind(log_1$f1,log_2$f1,log_3$f1,log_4$f1)))
  colnames(f1)[1] = "f1"
  f1_std = as.data.frame(colSds(rbind(log_1$f1,log_2$f1,log_3$f1,log_4$f1)))
  colnames(f1_std)[1] = "f1_std"
  valf1 = as.data.frame(colMeans(rbind(log_1$valf1,log_2$valf1,log_3$valf1,log_4$valf1)))
  colnames(valf1)[1] = "valf1"
  valf1_std = as.data.frame(colSds(rbind(log_1$valf1,log_2$valf1,log_3$valf1,log_4$valf1)))
  colnames(valf1_std)[1] = "valf1_std"
  print(valf1[75,1])
  print(valf1_std[75,1])
  
  plot(loss$loss~log_1$epoch, pch=20, cex=1, col=rgb(0,0,1,alpha=1), type="o", lwd=2,main=paste("Loss",name,sep=" "),xlab="epoch",ylab="loss", cex.axis=1, cex.lab=1, cex.main=1)
  arrows(log_1$epoch, loss$loss-loss_std$loss_std, log_1$epoch, loss$loss+loss_std$loss_std, length=0.05, angle=90, code=3, col=rgb(0,0,1,alpha=0.3) )
  points(valloss$valloss~log_1$epoch, pch=20, cex=1, col=rgb(1,0,0,alpha=1), type="o",lwd=2)
  arrows(log_1$epoch, valloss$valloss-valloss_std$valloss_std, log_1$epoch, valloss$valloss+valloss_std$valloss_std, length=0.05, angle=90, code=3, col=rgb(1,0,0,alpha=0.3) )
  legend("topright",inset=0.05,legend=c("Training", "Validation"), col=c("blue","red"), lty=1:1, cex=1,lwd=2)
  
  plot(acc$acc~log_1$epoch, pch=20, cex=1, col=rgb(0,0,1,alpha=1), type="o",main=paste("Accuracy",name,sep=" "),xlab="epoch",ylab="acc", cex.axis=2, cex.lab=2, cex.main=2)
  arrows(log_1$epoch, acc$acc-acc_std$acc_std, log_1$epoch, acc$acc+acc_std$acc_std, length=0.05, angle=90, code=3, col=rgb(0,0,1,alpha=0.3) )
  points(valacc$valacc~log_1$epoch, pch=20, cex=1, col=rgb(1,0,0,alpha=1), type="o")
  arrows(log_1$epoch, valacc$valacc-valacc_std$valacc_std, log_1$epoch, valacc$valacc+valacc_std$valacc_std, length=0.05, angle=90, code=3, col=rgb(1,0,0,alpha=0.3) )
  
  plot(f1$f1~log_1$epoch, pch=20, cex=1, col=rgb(0,0,1,alpha=1), type="o",main=paste("F1-Score",name,sep=" "),xlab="epoch",ylab="f1", cex.axis=2, cex.lab=2, cex.main=2)
  arrows(log_1$epoch, f1$f1-f1_std$f1_std, log_1$epoch, f1$f1+f1_std$f1_std, length=0.05, angle=90, code=3, col=rgb(0,0,1,alpha=0.3) )
  points(valf1$valf1~log_1$epoch, pch=20, cex=1, col=rgb(1,0,0,alpha=1), type="o")
  arrows(log_1$epoch, valf1$valf1-valf1_std$valf1_std, log_1$epoch, valf1$valf1+valf1_std$valf1_std, length=0.05, angle=90, code=3, col=rgb(1,0,0,alpha=0.3) )

}


########## PLOT ROC CURVES
dataList=list(c("clin_manq","roc",rgb(0,0,0,alpha=1)), c("multi_manq","roc",rgb(0,0,1,alpha=1)), c("multi_adapt_manq","roc",rgb(1,0,0,alpha=1)), c("clin_m_manq","roc",rgb(0.5,0.5,0.5,alpha=1)) ,c("multi_m_manq","roc",rgb(0,1,1,alpha=1)),c("multi_adapt_m_manq","roc",rgb(1,0,1,alpha=1)))
#1)dataList=list(c("clin_manq","roc",rgb(0,0,0,alpha=1),rgb(0,0,0,alpha=0.3)), c("multi_manq","roc",rgb(0,0,1,alpha=1),rgb(0,0,1,alpha=0.3)), c("multi_adapt_manq","roc",rgb(1,0,0,alpha=1),rgb(1,0,0,alpha=0.3)))
#2)dataList=list(c("multi_manq","roc",rgb(0,0,1,alpha=1)), c("multi_adapt_manq","roc",rgb(1,0,0,alpha=1)) ,c("multi_m_manq","roc",rgb(0,1,1,alpha=1)),c("multi_adapt_m_manq","roc",rgb(1,0,1,alpha=1)))
#3)dataList=list(c("clin_manq","roc",rgb(0,0,0,alpha=1)), c("multi_manq","roc",rgb(0,0,1,alpha=1)), c("clin_m_manq","roc",rgb(0.5,0.5,0.5,alpha=1)) ,c("multi_m_manq","roc",rgb(0,1,1,alpha=1)))
c=0
t=1
k=1
list_auc = c()
df = data.frame(matrix(ncol = 60, nrow = 0))
#x = c("test", "fold", "tpr")
#colnames(df) = x

for(name in dataList)
{
list_i = list("0","1","2","3","4","5")
for(i in list_i)
{
  
log3_1 = read.table(paste("~/Bureau/RScripts/",name[1],i,"/",name[2],"_0.csv",sep=""), header=TRUE, sep = ",")
log3_2 = read.table(paste("~/Bureau/RScripts/",name[1],i,"/",name[2],"_1.csv",sep=""), header=TRUE, sep = ",")
log3_3 = read.table(paste("~/Bureau/RScripts/",name[1],i,"/",name[2],"_2.csv",sep=""), header=TRUE, sep = ",")
log3_4 = read.table(paste("~/Bureau/RScripts/",name[1],i,"/",name[2],"_3.csv",sep=""), header=TRUE, sep = ",")

df[k,] = c(paste(name[1],i,sep=""),1,log3_1$tpr)
df[k+1,] = c(paste(name[1],i,sep=""),2,log3_2$tpr)
df[k+2,] = c(paste(name[1],i,sep=""),3,log3_3$tpr)
df[k+3,] = c(paste(name[1],i,sep=""),4,log3_4$tpr)

k = k+4

tpr = as.data.frame(colMeans(rbind(log3_1$tpr,log3_2$tpr,log3_3$tpr,log3_4$tpr)))
colnames(tpr)[1] = "tpr"
tpr_std = as.data.frame(colSds(rbind(log3_1$tpr,log3_2$tpr,log3_3$tpr,log3_4$tpr)))
colnames(tpr_std)[1] = "tpr_std"
fpr = as.data.frame(colMeans(rbind(log3_1$fpr,log3_2$fpr,log3_3$fpr,log3_4$fpr)))
colnames(fpr)[1] = "fpr"

if (i==0){pch=4}
if (i==1){pch=20}
if (i==2){pch=15}
if (i==3){pch=17}
if (i==4){pch=1}
if (i==5){pch=6}
if (c==0)
{
plot(tpr$tpr~fpr$fpr, pch=pch, cex=1, col=name[3], type="o", lwd = 2,xlab="FPR",ylab="TPR", cex.axis=1, cex.lab=1, cex.main=1)
c = 1
}
else
{
  points(tpr$tpr~fpr$fpr, pch=pch, cex=1, col=name[3], type="o", lwd = 2)
  
}
list_auc[t] = auc(tpr$tpr)
t = t+1
  #arrows(fpr$fpr, tpr$tpr-tpr_std$tpr_std, fpr$fpr, tpr$tpr+tpr_std$tpr_std, length=0.05, angle=90, code=3, col=name[4] )
}

}

#dataList=list(c("clin_manq","roc",rgb(0,1,1,alpha=1)), c("multi_manq","roc",rgb(0,0,1,alpha=1)), c("multi_adapt_manq","roc",rgb(0,0,0,alpha=1)) ,c("multi_m_manq","roc",rgb(1,0,0,alpha=1)),c("multi_adapt_m_manq","roc_adapt_manq",rgb(0,1,0,alpha=1)))

points(0:1, 0:1, cex=1, col=rgb(0,0,0,alpha=1), type="l",lty=2)
points(c(0.12,0.12), c(0,1), cex=1, col=rgb(0,0,0,alpha=1), type="l",lty=5)
#legend("bottomright",inset=0.05,legend=c("Clin","Multi", "MultiAdapt", "ClinM", "MultiM", "MultiAdaptM"), col=c(rgb(0,0,0,alpha=1), rgb(0,0,1,alpha=1), rgb(1,0,0,alpha=1), rgb(0.5,0.5,0.5,alpha=1), rgb(0,1,1,alpha=1), rgb(1,0,1,alpha=1) ), lty=1:1, cex=1)
legend("bottomright", title="Model           MissingData%", legend=c("Clin","Multi", "MultiAdapt", "ClinM", "MultiM", "MultiAdaptM","0","1/8","2/8","3/8","4/8","5/8"), col=c(rgb(0,0,0,alpha=1), rgb(0,0,1,alpha=1), rgb(1,0,0,alpha=1), rgb(0.5,0.5,0.5,alpha=1), rgb(0,1,1,alpha=1), rgb(1,0,1,alpha=1), rep("black",6) ), pch=c(NA,NA,NA,NA,NA,NA,4,20,15,17,1,6), lty=c(1,1,1,1,1,1,NA,NA,NA,NA,NA,NA), cex=1,ncol=2, lwd=2:2)

list_auc



