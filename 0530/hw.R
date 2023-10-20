library(data.table)
UKB<-fread("D:/SYSU/大创/ukb47147_head10.csv",sep=',',header=TRUE);
sex1<-"31-0.0";

UKB_sex <- UKB[,24-0.0];#21
UKB_sex;
head(UKB_sex);
a <- sum(UKB_sex);
plot_sex <- c(a,lengths(UKB_sex)-a);
barplot(plot_sex)



UKB_birthyear <- UKB[,25-0.0]#34
