#clasificacion binaria de keras
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

dataset_path = "/home/usco/Documents/sergio/sergio/dataset"


categories = ["cassava_bacterial_blight", "cassava_brown_streak_disease", "cassava_green_mottle", "cassava_mosaic_disease", "healthy"]

#de ellos
#input_shape = (224, 224, 3)
#nuestro (w-h)
#input_shape = (800, 600, 3)

num_skipped = 0
for category in categories:
    folder_path = os.path.join(dataset_path, category)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            # Eliminar imagen corrupta
            os.remove(fpath)

print("Se eliminaron %d imágenes" % num_skipped)






image_size = (224, 224)
batch_size = 8
num_classes = 5
num_channels = 3
input_shape = image_size + (num_channels,)

# Parameters
params = {'dim': image_size,
          'batch_size': batch_size,
          'n_classes': num_classes,
          'n_channels': num_channels,
          'shuffle': False}




# balancear las clases
import os
import pandas as pd
import numpy as np
import random
import glob

image_folder = "/home/usco/Documents/sergio/sergio/dataset"
cassava_bacterial_blight = os.path.join(image_folder, "cassava_bacterial_blight")
cassava_brown_streak_disease = os.path.join(image_folder, "cassava_brown_streak_disease")
cassava_green_mottle = os.path.join(image_folder, "cassava_green_mottle")
cassava_mosaic_disease = os.path.join(image_folder, "cassava_mosaic_disease")
healthy = os.path.join(image_folder, "healthy")

cassava_bacterial_blight_path = os.path.join(cassava_bacterial_blight, "*.jpg")
cassava_brown_streak_disease_path = os.path.join(cassava_brown_streak_disease, "*.jpg")
cassava_green_mottle_path = os.path.join(cassava_green_mottle, "*.jpg")
cassava_mosaic_disease_path = os.path.join(cassava_mosaic_disease, "*.jpg")
healthy_path = os.path.join(healthy, "*.jpg")


cassava_bacterial_blight_filenames = glob.glob(cassava_bacterial_blight_path)
cassava_brown_streak_disease_filenames = glob.glob(cassava_brown_streak_disease_path)
cassava_green_mottle_filenames = glob.glob(cassava_green_mottle_path)
cassava_mosaic_disease_filenames = glob.glob(cassava_mosaic_disease_path)
healthy_filenames = glob.glob(healthy_path)

print(len(cassava_bacterial_blight_filenames),len(cassava_brown_streak_disease_filenames),len(cassava_green_mottle_filenames),len(cassava_mosaic_disease_filenames),len(healthy_filenames))







# Determinar el número mínimo de imágenes en una categoría
min_samples = min(len(cassava_bacterial_blight_filenames),len(cassava_brown_streak_disease_filenames),len(cassava_green_mottle_filenames),len(cassava_mosaic_disease_filenames),len(healthy_filenames))

# Seleccionar aleatoriamente la misma cantidad de imágenes de cada categoría
random.shuffle(cassava_bacterial_blight_filenames)
random.shuffle(cassava_brown_streak_disease_filenames)
random.shuffle(cassava_green_mottle_filenames)
random.shuffle(cassava_mosaic_disease_filenames)
random.shuffle(healthy_filenames)

cassava_bacterial_blight_filenames = cassava_bacterial_blight_filenames[:min_samples]
cassava_brown_streak_disease_filenames = cassava_brown_streak_disease_filenames[:min_samples]
cassava_green_mottle_filenames = cassava_green_mottle_filenames[:min_samples]
cassava_mosaic_disease_filenames = cassava_mosaic_disease_filenames[:min_samples]
healthy_filenames = healthy_filenames[:min_samples]

# Crear un DataFrame con las rutas de las imágenes y sus etiquetas
df_cassava_bacterial_blight = pd.DataFrame({'filename': cassava_bacterial_blight_filenames, 'label': 0})
df_cassava_brown_streak_disease = pd.DataFrame({'filename': cassava_brown_streak_disease_filenames, 'label': 1})
df_cassava_green_mottle = pd.DataFrame({'filename': cassava_green_mottle_filenames, 'label': 2})
df_cassava_mosaic_disease = pd.DataFrame({'filename': cassava_mosaic_disease_filenames, 'label': 3})
df_healthy = pd.DataFrame({'filename': healthy_filenames, 'label': 4})

# print(len(df_cassava_bacterial_blight),len(df_cassava_brown_streak_disease),len(df_cassava_green_mottle),len(df_cassava_mosaic_disease),len(df_healthy))











num_iterations = 7
distribution = {"train":0.70, "val":0.15, "test":0.15}
len_train = int(min_samples * distribution["train"])
len_val = int(min_samples * distribution["val"])
len_test = int(min_samples * distribution["test"])

for i in range(num_iterations):
  # train dataset distribution
  start = i * len_val
  end = start + len_train
  df_train = pd.concat([
      df_cassava_bacterial_blight[start:end],
      df_cassava_brown_streak_disease[start:end],
      df_cassava_green_mottle[start:end],
      df_cassava_mosaic_disease[start:end],
      df_healthy[start:end]
  ])

  if len(df_train) < (len_train*num_classes)-1:
    start = 0
    end = len_train - int(len(df_train)/ num_classes)
    df_train2 = pd.concat([
      df_cassava_bacterial_blight[start:end],
      df_cassava_brown_streak_disease[start:end],
      df_cassava_green_mottle[start:end],
      df_cassava_mosaic_disease[start:end],
      df_healthy[start:end]
    ])

    df_train = pd.concat([df_train, df_train2])

  # print(i, start, end, len(df_train))

  start = end
  if start >= min_samples-1:
    start = 0
  end = start + len_val
  df_val = pd.concat([
      df_cassava_bacterial_blight[start:end],
      df_cassava_brown_streak_disease[start:end],
      df_cassava_green_mottle[start:end],
      df_cassava_mosaic_disease[start:end],
      df_healthy[start:end]
  ])
  # print(i, start, end, len(df_val))

  start = end
  if start >= min_samples-1:
    start = 0
  end = start + len_test
  df_test = pd.concat([
      df_cassava_bacterial_blight[start:end],
      df_cassava_brown_streak_disease[start:end],
      df_cassava_green_mottle[start:end],
      df_cassava_mosaic_disease[start:end],
      df_healthy[start:end]
  ])
  # print(i, start, end, len(df_test))


  train_filename = "train_ds_" + str(i) + ".csv"
  val_filename = "val_ds_" + str(i) + ".csv"
  test_filename = "test_ds_" + str(i) + ".csv"

  df_train = df_train.sample(frac=1)
  df_val = df_val.sample(frac=1)
  df_test = df_test.sample(frac=1)

  df_train.to_csv(train_filename)
  df_val.to_csv(val_filename)
  df_test.to_csv(test_filename)

  # print("-"*60)
  # print()

for i in range(num_iterations):
  train_filename = "train_ds_" + str(i) + ".csv"
  val_filename = "val_ds_" + str(i) + ".csv"
  test_filename = "test_ds_" + str(i) + ".csv"

  df_train = pd.read_csv(train_filename)
  df_val = pd.read_csv(val_filename)
  df_test = pd.read_csv(test_filename)

  print(df_train.groupby(["label"])["label"].count())
  print(df_val.groupby(["label"])["label"].count())
  print(df_test.groupby(["label"])["label"].count())
  # print("-"*60)
  # print()



for i in range(num_iterations):
  train_filename = "train_ds_" + str(i) + ".csv"
  val_filename = "val_ds_" + str(i) + ".csv"
  test_filename = "test_ds_" + str(i) + ".csv"

  df_train = pd.read_csv(train_filename)
  df_val = pd.read_csv(val_filename)
  df_test = pd.read_csv(test_filename)

  print(df_train.groupby(["label"])["label"].count())
  print(df_val.groupby(["label"])["label"].count())
  print(df_test.groupby(["label"])["label"].count())
  # print("-"*60)
  # print()






i = 0
train_filename = "train_ds_" + str(i) + ".csv"
val_filename = "val_ds_" + str(i) + ".csv"
test_filename = "test_ds_" + str(i) + ".csv"

df_train = pd.read_csv(train_filename)
df_val = pd.read_csv(val_filename)
df_test = pd.read_csv(test_filename)



partition = {}
partition["train"] =  list(df_train["filename"])
partition["val"] =  list(df_val["filename"])
partition["test"] =  list(df_test["filename"])



labels = {}
df_all = pd.concat([df_train, df_val, df_test])
for index, row in df_all.iterrows():
  filename = row["filename"]
  label = row["label"]
  labels[filename] = label
# print(labels)






# Importing Image class from PIL module
from PIL import Image

def get_image(image_filename):
    
  # Opens a image in RGB mode
  im1 = Image.open(image_filename).convert("RGB")

  im1 = im1.resize(image_size)
  # print(type(im1))
  image = np.asarray(im1)
  image = np.array(image, dtype='float32')
  image = image /255.0
  # print(image.shape)
  return image



import numpy as np
import keras





class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=8, dim=(224,224), n_channels=3,
                 n_classes=5, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            #X[i,] = np.load('data/' + ID + '.npy')
            image_filename = os.path.join(image_folder, ID)
            X[i,] = get_image(image_filename)

            # Store class
            y[i] = self.labels[ID]

            #print(image_filename, y[i])

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
    
    
    
    
# Generators
train_generator = DataGenerator(partition['train'], labels, **params)
val_generator = DataGenerator(partition['val'], labels, **params)
test_generator = DataGenerator(partition['test'], labels, **params)



epochs= 100
callbacks = [
    keras.callbacks.EarlyStopping(monitor='loss', patience=3)
]



import keras_tuner as kt



def model_builder(hp):
  hp_model_type = hp.Choice(
    "model_type",["DenseNet121","EfficientNetB0","ResNet101V2"], default="DenseNet121"
)
    # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, or 0.0001
  hp_learning_rate = hp.Choice(
      'learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5]
      )

  #optimizer1=keras.optimizers.Adam(learning_rate=hp_learning_rate)
  #optimizer2=keras.optimizers.SGD(learning_rate=hp_learning_rate)
  #optimizer3=keras.optimizers.RMSprop(learning_rate=hp_learning_rate)
  hp_optimizer = hp.Choice(
      "optimizer", ["Adam","SGD","RMSprop"], default = "Adam"
  )

  with hp.conditional_scope("model_type", ["DenseNet121"]):
    if hp_model_type == "DenseNet121":
      model = tf.keras.applications.DenseNet121(
          include_top=True,
          weights=None,
          input_shape=input_shape,
          classes=num_classes,
          classifier_activation="softmax",
      )

  with hp.conditional_scope("model_type", ["EfficientNetB0"]):
    if hp_model_type == "EfficientNetB0":
      model = tf.keras.applications.EfficientNetB0(

          include_top=True,
          weights=None,
          input_shape=input_shape,
          classes=num_classes,
          classifier_activation="softmax",
      )

  with hp.conditional_scope("model_type", ["ResNet101V2"]):
    if hp_model_type == "ResNet101V2":
      model = tf.keras.applications.ResNet101V2(

          include_top=True,
          weights=None,
          input_shape=input_shape,
          classes=num_classes,
          classifier_activation="softmax",
      )

  with hp.conditional_scope("optimizer", ["Adam"]):
    if hp_optimizer == "Adam":
      optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate)

  with hp.conditional_scope("optimizer", ["SGD"]):
    if hp_optimizer == "SGD":
      optimizer=keras.optimizers.SGD(learning_rate=hp_learning_rate)

  with hp.conditional_scope("optimizer", ["RMSprop"]):
    if hp_optimizer == "RMSprop":
      optimizer=keras.optimizers.RMSprop(learning_rate=hp_learning_rate)




#optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
  model.compile(
    optimizer=hp_optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"],
  )

  return model





tuner = kt.Hyperband(
    model_builder,
    objective='val_accuracy',
    max_epochs=100,
    factor=3,
    overwrite=False,
    directory='/home/usco/Documents/sergio/sergio/directory2',
    project_name='intro_to_kt'
    )




tuner.search(
    train_generator,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_generator,
    )

# import keras_tuner as kt


# Get the optimal hyperparameters
# best_hps=tuner.get_best_hyperparameters(num_trials=3)
best_hps=tuner.get_best_hyperparameters(num_trials=31)
for i in best_hps:
    
    print(f"""
    The hyperparameter search is complete. 
    , optimal learning rate for the optimizer is {i.get('learning_rate')}.
    , optimal model name is {i.get('model_type')}.
    , optimal optimizer name is {i.get('optimizer')}.
    """)




# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=3)


print(best_hps)


# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
model = tuner.hypermodel.build(best_hps)
#history = model.fit(img_train, label_train, epochs=50, validation_split=0.2)

history = model.fit(train_generator,
epochs=100,
callbacks=callbacks,
validation_data=val_generator,
initial_epoch=0
)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))


model_save_path = "/home/usco/Documents/sergio/sergio/modelsergiog2v2.h5"
model.save(model_save_path)



import matplotlib.pyplot as plt


plt.xlabel("# epoch")


plt.ylabel("Loss Magnitude")


plt.plot(history.history["loss"])











































