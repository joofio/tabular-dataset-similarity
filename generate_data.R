library("synthpop")

df<-read.csv("/Users/joaoalmeida/OneDrive/Pessoal/HEADS/05. TESE/8.0 dataset similarity/real_data_testing.csv")
df$X<-NULL
myseed <- 20190110
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
write.csv(synth.obj$syn,"/Users/joaoalmeida/OneDrive/Pessoal/HEADS/05. TESE/8.0 dataset similarity/synth_pop_2.csv")
