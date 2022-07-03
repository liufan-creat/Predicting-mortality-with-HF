library(keras)
library(mice)
library(pROC)

data <- read.csv('~/MIMIC_III_heart.csv',header = T,row.names = 2,stringsAsFactors = F)
imp <- mice(data, m=10,meth =  "pmm")

fit <- with(imp, lm(PH~PCO2+Lactic.acid))
pooled <- pool(fit)
pool.r.squared(fit)
summary(fit)

fit <- with(imp, lm(Leucocyte~Neutrophils))
pooled <- pool(fit)
pool.r.squared(fit)
summary(fit)

exp <- complete(imp,3)
exp_scale <- apply(exp[,3:50], 2, scale) 

train <- exp_scale[which(exp$group =='1'),]
 
test <- exp_scale[which(exp$group =='2'),]
 


train_sample <- exp[which(exp$group =='1'),]
train_res <-train_sample$outcome
train_res <- to_categorical(train_res,2)

test_sample <- exp[which(exp$group =='2'),]
test_res <-test_sample$outcome
test_res <- to_categorical(test_res,2)


model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 48, activation = 'relu', input_shape = c(48)) %>% 
  layer_dropout(rate = 0.2) %>%  
  layer_dense(units = 2, activation = 'softmax')


model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = c('accuracy') )

history <- model %>% fit(train, train_res, 
                         epochs = 100, batch_size = 120, 
                         validation_split = 0.2 )

plot(history)

model %>%evaluate(test, test_res)
model %>%evaluate(train, train_res)

a <-  model %>%predict(train) 
roc <- roc(train_res[,1], a[,1],percent=T,smooth=T , auc=T)
print(roc)

plot.roc(roc,print.auc=T,
         auc.polygon=T,auc.polygon.col='skyblue')

### write
weight <- get_weights(model)
save_model_hdf5(model,"model_1.h5")
new_model <- load_model_hdf5('model_1.h5')

## read 
example1 <- read.csv('example.csv',stringsAsFactors = F,row.names = 1)
## scale
for (i in 1:length(example1[,1])) {
  id <- rownames(example1)[i]
  mean <- mean(exp[,id])
  var <- var(exp[,id])
  example1$scale[i] <- scale(example1[i,1],center = mean,scale = var)
}
## call
new_model %>%predict(t(example1$scale))[2]
  


