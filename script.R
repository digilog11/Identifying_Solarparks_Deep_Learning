library(keras)
library(tensorflow)
library(tfdatasets)
library(purrr)
library(ggplot2)
library(rsample)
library(stars)
library(raster)
library(reticulate)
library(mapview)



### build model

## we start with the "contratcing path"##
# input
input_tensor <- layer_input(shape = c(128,128,4))

#conv block 1
unet_tensor <- layer_conv_2d(input_tensor,filters = 64,kernel_size = c(3,3), padding = "same",activation = "relu")
conc_tensor2 <- layer_conv_2d(unet_tensor,filters = 64,kernel_size = c(3,3), padding = "same",activation = "relu")
unet_tensor <- layer_max_pooling_2d(conc_tensor2)

#conv block 2
unet_tensor <- layer_conv_2d(unet_tensor,filters = 128,kernel_size = c(3,3), padding = "same",activation = "relu")
conc_tensor1 <- layer_conv_2d(unet_tensor,filters = 128,kernel_size = c(3,3), padding = "same",activation = "relu")
unet_tensor <- layer_max_pooling_2d(conc_tensor1)

#conv block 3
unet_tensor <- layer_conv_2d(unet_tensor,filters = 256,kernel_size = c(3,3), padding = "same",activation = "relu")
conc_tensor1 <- layer_conv_2d(unet_tensor,filters = 256,kernel_size = c(3,3), padding = "same",activation = "relu")
unet_tensor <- layer_max_pooling_2d(conc_tensor1)

#conv block 4
unet_tensor <- layer_conv_2d(unet_tensor,filters = 512,kernel_size = c(3,3), padding = "same",activation = "relu")
conc_tensor1 <- layer_conv_2d(unet_tensor,filters = 512,kernel_size = c(3,3), padding = "same",activation = "relu")
unet_tensor <- layer_max_pooling_2d(conc_tensor1)

#"bottom curve" of unet
unet_tensor <- layer_conv_2d(unet_tensor,filters = 1024,kernel_size = c(3,3), padding = "same",activation = "relu")
unet_tensor <- layer_conv_2d(unet_tensor,filters = 1024,kernel_size = c(3,3), padding = "same",activation = "relu")

##  this is where the expanding path begins ##

# upsampling block 1
unet_tensor <- layer_conv_2d_transpose(unet_tensor,filters = 512,kernel_size = c(2,2),strides = 2,padding = "same") 
unet_tensor <- layer_concatenate(list(conc_tensor1,unet_tensor))
unet_tensor <- layer_conv_2d(unet_tensor, filters = 512, kernel_size = c(3,3),padding = "same", activation = "relu")
unet_tensor <- layer_conv_2d(unet_tensor, filters = 512, kernel_size = c(3,3),padding = "same", activation = "relu")

# upsampling block 2
unet_tensor <- layer_conv_2d_transpose(unet_tensor,filters = 256,kernel_size = c(2,2),strides = 2,padding = "same") 
unet_tensor <- layer_concatenate(list(conc_tensor1,unet_tensor))
unet_tensor <- layer_conv_2d(unet_tensor, filters = 256, kernel_size = c(3,3),padding = "same", activation = "relu")
unet_tensor <- layer_conv_2d(unet_tensor, filters = 256, kernel_size = c(3,3),padding = "same", activation = "relu")

# upsampling block 3
unet_tensor <- layer_conv_2d_transpose(unet_tensor,filters = 128,kernel_size = c(2,2),strides = 2,padding = "same")
unet_tensor <- layer_concatenate(list(conc_tensor2,unet_tensor))
unet_tensor <- layer_conv_2d(unet_tensor, filters = 128, kernel_size = c(3,3),padding = "same", activation = "relu")
unet_tensor <- layer_conv_2d(unet_tensor, filters = 128, kernel_size = c(3,3),padding = "same", activation = "relu")

# upsampling block 4
unet_tensor <- layer_conv_2d_transpose(unet_tensor,filters = 64,kernel_size = c(2,2),strides = 2,padding = "same")
unet_tensor <- layer_concatenate(list(conc_tensor2,unet_tensor))
unet_tensor <- layer_conv_2d(unet_tensor, filters = 64, kernel_size = c(3,3),padding = "same", activation = "relu")
unet_tensor <- layer_conv_2d(unet_tensor, filters = 64, kernel_size = c(3,3),padding = "same", activation = "relu")

# output
unet_tensor <- layer_conv_2d(unet_tensor,filters = 1,kernel_size = 1, activation = "sigmoid")

# combine final unet_tensor (carrying all the transformations applied through the layers) 
# with input_tensor to create model

unet_model <- keras_model(inputs = input_tensor, outputs = unet_tensor)



### functions for preparing data

read_tif <- function(f,mask=FALSE) {
  out = array(NA)
  out = unclass(read_stars(f))[[1]]
  if(mask==T){
    dim(out) <- c(dim(out),1)
  }
  return(out)
}

dl_prepare_data_tif <- function(files, train, predict=FALSE, subsets_path=NULL, model_input_shape = c(448,448), batch_size = 10L) {
  
  if (!predict){
    
    #function for random change of saturation,brightness and hue, will be used as part of the augmentation
    spectral_augmentation <- function(img) {
      img %>% 
        tf$image$random_brightness(max_delta = 0.3) %>% 
        tf$image$random_contrast(lower = 0.5, upper = 0.7) %>% 
        #tf$image$random_saturation(lower = 0.5, upper = 0.7) %>%  --> not supported for >3 bands - you can uncomment in case you use only 3band images
        # make sure we still are between 0 and 1
        tf$clip_by_value(0, 1) 
    }
    
    #create a tf_dataset from the first two coloumns of data.frame (ignoring area number used for splitting during data preparation),
    #right now still containing only paths to images 
    dataset <- tensor_slices_dataset(files[,1:2])
    
    #the following (replacing tf$image$decode_jpeg by the custom read_tif function) doesn't work, since read_tif cannot be used with dataset_map -> dl_prepare_data_tif therefore expects a data.frame with arrays (i.e. images already loaded)
    #dataset <- dataset_map(dataset, function(.x) list_modify(.x,
    #                                                         img = read_tif(.x$img)/10000,
    #                                                         mask = read_tif(.x$mask)#[1,,,][,,1,drop=FALSE]
    #)) 
    
    #convert to float32:
    #for each record in dataset, both its list items are modyfied by the result of applying convert_image_dtype to them
    dataset <- dataset_map(dataset, function(.x) list_modify(.x,
                                                             img = tf$image$convert_image_dtype(.x$img, dtype = tf$float64),
                                                             mask = tf$image$convert_image_dtype(.x$mask, dtype = tf$float64)
    )) 
    
    #resize:
    #for each record in dataset, both its list items are modified by the results of applying resize to them 
    dataset <- 
      dataset_map(dataset, function(.x) 
        list_modify(.x, img = tf$image$resize(.x$img, size = shape(model_input_shape[1], model_input_shape[2])),
                    mask = tf$image$resize(.x$mask, size = shape(model_input_shape[1], model_input_shape[2]))))
    
    # data augmentation performed on training set only
    if (train) {
      
      #augmentation 1: flip left right, including random change of saturation, brightness and contrast
      
      #for each record in dataset, only the img item is modified by the result of applying spectral_augmentation to it
      augmentation <- dataset_map(dataset, function(.x) list_modify(.x,
                                                                    img = spectral_augmentation(.x$img)
      ))
      #...as opposed to this, flipping is applied to img and mask of each record
      augmentation <- dataset_map(augmentation, function(.x) list_modify(.x,
                                                                         img = tf$image$flip_left_right(.x$img),
                                                                         mask = tf$image$flip_left_right(.x$mask)
      ))
      dataset_augmented <- dataset_concatenate(dataset,augmentation)
      
      #augmentation 2: flip up down, including random change of saturation, brightness and contrast
      augmentation <- dataset_map(dataset, function(.x) list_modify(.x,
                                                                    img = spectral_augmentation(.x$img)
      ))
      augmentation <- dataset_map(augmentation, function(.x) list_modify(.x,
                                                                         img = tf$image$flip_up_down(.x$img),
                                                                         mask = tf$image$flip_up_down(.x$mask)
      ))
      dataset_augmented <- dataset_concatenate(dataset_augmented,augmentation)
      
      #augmentation 3: flip left right AND up down, including random change of saturation, brightness and contrast
      augmentation <- dataset_map(dataset, function(.x) list_modify(.x,
                                                                    img = spectral_augmentation(.x$img)
      ))
      augmentation <- dataset_map(augmentation, function(.x) list_modify(.x,
                                                                         img = tf$image$flip_left_right(.x$img),
                                                                         mask = tf$image$flip_left_right(.x$mask)
      ))
      augmentation <- dataset_map(augmentation, function(.x) list_modify(.x,
                                                                         img = tf$image$flip_up_down(.x$img),
                                                                         mask = tf$image$flip_up_down(.x$mask)
      ))
      dataset_augmented <- dataset_concatenate(dataset_augmented,augmentation)
      
    }
    
    # shuffling on training set only
    if (train) {
      dataset <- dataset_shuffle(dataset_augmented, buffer_size = batch_size*128)
    }
    
    # train in batches; batch size might need to be adapted depending on
    # available memory
    dataset <- dataset_batch(dataset, batch_size)
    
    # output needs to be unnamed
    dataset <-  dataset_map(dataset, unname) 
    
  }else{
    #make sure subsets are read in in correct order so that they can later be reassambled correctly
    #needs files to be named accordingly (only number)
    o <- order(as.numeric(tools::file_path_sans_ext(basename(list.files(subsets_path)))))
    subset_list <- list.files(subsets_path, full.names = T)[o]
    
    dataset <- tensor_slices_dataset(subset_list)
    #dataset <- dataset_map(dataset, function(.x) tf$image$decode_jpeg(tf$io$read_file(.x))) 
    dataset <- dataset_map(dataset, function(.x) tf$image$convert_image_dtype(.x, dtype = tf$float32)) 
    dataset <- dataset_map(dataset, function(.x) tf$image$resize(.x, size = shape(model_input_shape[1], model_input_shape[2]))) 
    dataset <- dataset_batch(dataset, batch_size)
    dataset <-  dataset_map(dataset, unname)
    
  }
  
}

dl_prepare_data_tif2 <- function(files, model_input_shape = c(448,448), batch_size = 10L) {
  
    #create a tf_dataset from the first two coloumns of data.frame (ignoring area number used for splitting during data preparation),
    #right now still containing only paths to images 
    dataset <- tensor_slices_dataset(files)
    
    #convert to float32:
    #for each record in dataset, both its list items are modyfied by the result of applying convert_image_dtype to them
    dataset <- dataset_map(dataset, function(.x) list_modify(.x,
       img = tf$image$convert_image_dtype(.x$img, dtype = tf$float64)
    )) 
    
    #resize:
    #for each record in dataset, both its list items are modified by the results of applying resize to them 
    dataset <- 
      dataset_map(dataset, function(.x) 
        list_modify(.x, img = tf$image$resize(.x$img, size = shape(model_input_shape[1], model_input_shape[2]))))
    
    # train in batches; batch size might need to be adapted depending on
    # available memory
    dataset <- dataset_batch(dataset, batch_size)
    
    # output needs to be unnamed
    dataset <-  dataset_map(dataset, unname) 
}



### load data

files <- data.frame(
  img = list.files("./kacheln128_3/img", full.names = TRUE, pattern = "*.tif"),
  mask = list.files("./kacheln128_3/mask", full.names = TRUE, pattern = "*.tif")
)

files$img <- lapply(files$img, read_tif)
files$img <- lapply(files$img, function(x){x/10000}) 
files$mask <- lapply(files$mask, read_tif, TRUE)

# split the data into training and validation datasets

files <- initial_split(files, prop = 0.8)

# prepare data for training

training_dataset <- dl_prepare_data_tif(training(files),train = TRUE,model_input_shape = c(128,128),batch_size = 10L)
validation_dataset <- dl_prepare_data_tif(testing(files),train = FALSE,model_input_shape = c(128,128),batch_size = 10L)

# get all tensors through the python iterator
training_tensors <- training_dataset%>%as_iterator()%>%iterate()

# how many tensors?
length(training_tensors)



### train

compile(
  unet_model,
  optimizer = optimizer_rmsprop(lr = 1e-5),
  loss = "binary_crossentropy",
  metrics = c(metric_binary_accuracy)
)


diagnostics <- fit(unet_model,
                   training_dataset,
                   epochs = 45,
                   validation_data = validation_dataset)

plot(diagnostics)



### save model

save_model_hdf5(unet_model,filepath = "./unet_model.h5")

### load model

unet_model <- load_model_hdf5("./unet_model.h5")



### plot mask, img and pred of a sample tile from validation dataset

sample <- floor(runif(n = 1,min = 1,max = 100))

mask <- magick::image_read(as.raster(matrix(testing(files)[[sample,2]], nrow=128, ncol=128)))
plot(mask)

pred <- magick::image_read(as.raster(predict(object = unet_model,validation_dataset)[sample,,,]))
plot(pred)

x = predict(object = unet_model,validation_dataset)[sample,,,]

img = magick::image_normalize(magick::image_read(testing(files)[[sample,1]][,,3:1]))
plot(img)

out <- magick::image_append(c(
  magick::image_append(mask, stack = TRUE),
  magick::image_append(img, stack = TRUE), 
  magick::image_append(pred, stack = TRUE)
))

plot(out)




### test model

## Cestas

cestas <- data.frame(
  img = list.files("./cestas/img", full.names = TRUE, pattern = "*.tif")
)

cestas$img <- lapply(cestas$img, read_tif)
cestas$img <- lapply(cestas$img, function(x){x/10000}) 

test_dataset <- dl_prepare_data_tif2(cestas, model_input_shape = c(128,128),batch_size = 5L)

# create pred and save as tif-file
number = 4 # number of tiles 

for (i in 1:number){
  pred = raster(predict(object = unet_model,test_dataset)[i,,,])
  plot(pred)
  name = paste("cestas_raster/cestas_raster_", i, ".tif", sep="")
  writeRaster(pred, name, format="GTiff")
}

## Benban

benban <- data.frame(
  img = list.files("./benban", full.names = TRUE, pattern = "*.tif")
)

benban$img <- lapply(benban$img, read_tif)
benban$img <- lapply(benban$img, function(x){x/10000}) 

test_dataset <- dl_prepare_data_tif2(benban, model_input_shape = c(128,128),batch_size = 5L)

# create pred and save as tif-file
number = 4 # number of tiles 

for (i in 1:number){
  pred = raster(predict(object = unet_model,test_dataset)[i,,,])
  plot(pred)
  name = paste("benban_pred/benban_raster_", i, ".tif", sep="")
  writeRaster(pred, name, format="GTiff")
}
