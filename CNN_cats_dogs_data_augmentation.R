# Extract the zip file in this folder 

base_dir <- '~/cats_and_dogs_small'

train_dir <- file.path(base_dir, 'train')
validation_dir <- file.path(base_dir , 'validation')
test_dir <- file.path(base_dir, 'test')





#Using image data generator to read images from directories

#### Data augmentation and reducing overfitting 

datagen <- image_data_generator(
  rescale = 1/255 ,
  rotation_range = 40 , 
  width_shift_range = 0.2 , 
  height_shift_range = 0.2 ,
  shear_range = 0.2 ,
  zoom_range = 0.2 ,
  horizontal_flip = TRUE ,
  fill_mode = 'nearest'
)


test_datagen <- image_data_generator(rescale = 1/255)





#Infinite loop for flowing images in batches of 20

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(150,150),
  batch_size = 32,
  class_mode = 'binary' ,
  
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  test_datagen,
  target_size = c(150,150) ,
  batch_size = 20 , 
  class_mode = 'binary' 
  
)

# Model Architecture

model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32 , kernel_size = c(3,3) , activation = 'relu' , 
                input_shape = c(150,150,3)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 64 , kernel_size = c(3,3) , activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 128 , kernel_size = c(3,3) , activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3,3) , activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units = 512 , activation = 'relu') %>%
  layer_dense(units = 1 , activation = 'sigmoid')




model


#Compiling the model


model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_rmsprop(1e-4), ## Adams  optimizer and rmsprop are used for image recognition mainly 
  metrics = c('acc')
)




#Fitting the model with fit generator function 
##steps per epoch controls the infinte loop (in this case we have 20 images per epoch so 20x100 = 2000 imgaes which are presnt in the train folder to be fed into the data )

history <- model %>% fit_generator (train_generator,
                                    steps_per_epoch = 100 ,
                                    epochs = 50,
                                    validation_data = validation_generator ,
                                    validation_steps = 50 , 
                                    
)

