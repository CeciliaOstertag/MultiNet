library(corrplot)

df1 = read.table("/home/cecilia/Bureau/These/scripts/ADNIMERGE.csv", header=TRUE, sep = ",",na.strings=c(""," "))
#df2 = read.table("/home/cecilia/subjects.csv", header=FALSE, sep = ",")
df2 = as.data.frame(unique(df1$PTID))
colnames(df2)[1] = "V1"
dx = df1$DX
df = df1[,15:34]
df[,21] = unclass(dx)
names(df)[21]="DX"
#write.table(x=df,file="/home/cecilia/Bureau/RScripts/attributes.csv",sep=",",row.names=FALSE)  #then remove > and < signs in columns
df = read.table("/home/cecilia/Bureau/RScripts/attributes.csv", header=TRUE, sep = ",")
df[,22] = df1$AGE
names(df)[22]="AGE"
df[,23] = df1$PTID
df[,24] = df1$VISCODE
infos<-merge(df,df2,by.x="V23",by.y="V1")
infos = infos[infos$V24 %in% c("bl","m06","m12"),]
infos = infos[,2:23]
summary(infos)
na_count <-sapply(infos, function(y) sum(length(which(is.na(y)))))
na_count <- data.frame(na_count)

data<-merge(df,df2,by.x="V23",by.y="V1")
data = data[,2:23]
M = cor(data,use="pairwise.complete.obs")
corrplot(M, method="number", type="lower",tl.col="black", tl.srt=45)
data2 = data
data2[is.na(data)] <- 0
M2 = cor(data2,use="pairwise.complete.obs")
corrplot(M2, method="color", type="lower",tl.col="black", tl.srt=45)

#correlation accross all follow-up
corrplot(M, method="color", type="lower",tl.col="black", tl.srt=45)

data<-merge(df1,df2,by.x="PTID",by.y="V1")
#check number of NA for selected time points : BL, M06, M12
data2 = data[data$VISCODE %in% c("bl","m06","m12"),]
data2=data2[,15:34]
summary(data2)

data=data[,15:34]
write.table(x=data,file="attributes.csv",sep=",",row.names=FALSE)  #then remove > and < signs in columns
data = read.table("attributes.csv", header=TRUE, sep = ",")
data[is.na(data)] <- 0
M = cor(data,use="pairwise.complete.obs")

library(corrplot)
#correlation accross all follow-up
corrplot(M, method="color", type="lower",order="hclust",tl.col="black", tl.srt=45)
