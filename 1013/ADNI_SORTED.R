setwd("C:/Users/Lucille/Desktop/dachuang/biostat/materials/1013")
library(readr)
library(dplyr)
library(tidyr)
ADNIMERGE_07Oct2023 <- read_csv("C:/Users/Lucille/Desktop/dachuang/biostat/ADNIMERGE_07Oct2023.csv")
ADNI<-ADNIMERGE_07Oct2023
ADNI_sorted_longi <- arrange(ADNI, RID)


ADNI_sorted_panel <- arrange(ADNI,VISCODE)
write.table(ADNI_sorted_longi,"ADNI_sorted_longi.csv",row.names=FALSE,col.names=TRUE,sep=",")
write.table(ADNI_sorted_panel,"ADNI_sorted_panel.csv",row.names=FALSE,col.names=TRUE,sep=",")


unique_visitcodes <- unique(ADNI_sorted_panel_ADNI1$VISCODE)
print(unique_visitcodes)
ADNI_sorted_panel_bl<-subset(ADNI_sorted_panel_ADNI1,VISCODE=="bl")
ADNI_sorted_panel_m36<-subset(ADNI_sorted_panel_ADNI1,VISCODE=="m36")
ADNI_sorted_panel_m60<-subset(ADNI_sorted_panel,VISCODE=="m60")
ADNI_sorted_panel_m66<-subset(ADNI_sorted_panel,VISCODE=="m66")
ADNI_sorted_panel_m126<-subset(ADNI_sorted_panel,VISCODE=="m126")


write.table(ADNI_sorted_panel_ADNI1,"ADNI_sorted_panel_ADNI1.csv",row.names=FALSE,col.names=TRUE,sep=",")
write.table(ADNI_sorted_panel_ADNIGO,"ADNI_sorted_panel_ADNIGO.csv",row.names=FALSE,col.names=TRUE,sep=",")
write.table(ADNI_sorted_panel_ADNI2,"ADNI_sorted_panel_ADNI2.csv",row.names=FALSE,col.names=TRUE,sep=",")
write.table(ADNI_sorted_panel_ADNI3,"ADNI_sorted_panel_ADNI3.csv",row.names=FALSE,col.names=TRUE,sep=",")


library(ggplot2)
ggplot(ADNI_sorted_panel_bl, aes(x = DX_bl )) +
  geom_bar() +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.5)

ggplot(ADNI_sorted_panel_m36, aes(x = DX_bl )) +
  geom_bar() +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.5)

ggplot(ADNI_sorted_panel_ADNI2, aes(x = DX_bl )) +
  geom_bar() +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.5)

ggplot(ADNI_sorted_panel_ADNI3, aes(x = DX_bl )) +
  geom_bar() +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.5)