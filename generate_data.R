library("synthpop")
myseed <- 20190110

df<-read.csv("/Users/joaoalmeida/OneDrive/HEADS/05. TESE/8.0 dataset similarity/tabular-dataset-similarity/real_data_testing.csv")
df$X<-NULL
##['ca', 'cp', 'exang', 'fbs', 'num', 'restecg', 'sex', 'slope', 'thal'], dtype='object')

df$restecg <- as.factor(df$restecg)
df$ca <- as.factor(df$ca)
df$cp <- as.factor(df$cp)
df$exang <- as.factor(df$exang)
df$fbs <- as.factor(df$fbs)
df$sex <- as.factor(df$sex)
df$slope <- as.factor(df$slope)
df$thal <- as.factor(df$thal)

df$thal

synth.obj <- syn(df, seed = myseed)

synth.obj
df$restecg


compare(synth.obj, df, nrow = 3, ncol = 4)
table(synth.obj$syn[,c("cp", "fbs")])
table(df[,c("cp", "fbs")])

synth.obj$syn$restecg
write.csv(synth.obj$syn,"/Users/joaoalmeida/OneDrive/HEADS/05. TESE/8.0 dataset similarity/tabular-dataset-similarity/synth_pop_1.csv")

### 2 

df2<-read.csv("/Users/joaoalmeida/OneDrive/HEADS/05. TESE/8.0 dataset similarity/tabular-dataset-similarity/real_data2_testing.csv")
df2$X<-NULL

df2
df2[sapply(df2, is.character)] <- lapply(df2[sapply(df2, is.character)], factor)
df2
str(df2)

synth.obj2 <- syn(df2, seed = myseed)

synth.obj2


compare(synth.obj2, df2, nrow = 3, ncol = 4)
table(synth.obj2$syn[,c("Uniformity_of_Cell_Size", "Uniformity_of_Cell_Shape")])
table(df2[,c("Uniformity_of_Cell_Size", "Uniformity_of_Cell_Shape")])

synth.obj2$syn$Uniformity_of_Cell_Size
write.csv(synth.obj2$syn,"/Users/joaoalmeida/OneDrive/HEADS/05. TESE/8.0 dataset similarity/tabular-dataset-similarity/synth_pop_2.csv")


### 3 

df3<-read.csv("/Users/joaoalmeida/OneDrive/HEADS/05. TESE/8.0 dataset similarity/tabular-dataset-similarity/real_data3_testing.csv")
df3$X<-NULL

df3
df3[sapply(df3, is.character)] <- lapply(df3[sapply(df3, is.character)], factor)



synth.obj3 <- syn(df3, seed = myseed)

synth.obj3


compare(synth.obj3, df3, nrow = 3, ncol = 4)
table(synth.obj3$syn[,c("alkphos", "sgpt")])
table(df3[,c("alkphos", "sgpt")])

write.csv(synth.obj3$syn,"/Users/joaoalmeida/OneDrive/HEADS/05. TESE/8.0 dataset similarity/tabular-dataset-similarity/synth_pop_3.csv")


### 4

df4<-read.csv("/Users/joaoalmeida/OneDrive/HEADS/05. TESE/8.0 dataset similarity/tabular-dataset-similarity/real_data4_testing.csv")
df4$X<-NULL

df4
df4[sapply(df4, is.character)] <- lapply(df4[sapply(df4, is.character)], factor)



synth.obj4 <- syn(df4, seed = myseed)

synth.obj4


compare(synth.obj4, df4, nrow = 3, ncol = 4)
table(synth.obj4$syn[,c("T3", "TSTRI")])
table(df4[,c("T3", "TSTRI")])

write.csv(synth.obj4$syn,"/Users/joaoalmeida/OneDrive/HEADS/05. TESE/8.0 dataset similarity/tabular-dataset-similarity/synth_pop_4.csv")


### 5

df5<-read.csv("/Users/joaoalmeida/OneDrive/HEADS/05. TESE/8.0 dataset similarity/tabular-dataset-similarity/real_data5_testing.csv")
df5$X<-NULL

df5
df5[sapply(df5, is.character)] <- lapply(df5[sapply(df5, is.character)], factor)



synth.obj5 <- syn(df5, seed = myseed)

synth.obj5


compare(synth.obj5, df5, nrow = 3, ncol = 4)
table(synth.obj5$syn[,c("exocytosis", "parakeratosis")])
table(df5[,c("exocytosis", "parakeratosis")])

write.csv(synth.obj5$syn,"/Users/joaoalmeida/OneDrive/HEADS/05. TESE/8.0 dataset similarity/tabular-dataset-similarity/synth_pop_5.csv")

