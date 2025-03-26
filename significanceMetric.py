from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import SGD, Adam, Nadam
from tensorflow.keras import utils
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, Callback

'''
The following class definition is the custom significance class I use to choose the best training epoch, by selecting the highest validation point, as well as an agreement of less than 1% between its values for the 
training and validation datasets. I've also used it to understand if a MLP model is already very huge and unnecessary for the task of separating VBS from background
'''
@tf.keras.utils.register_keras_serializable()
class globalSignificance(tf.keras.metrics.Metric):

  def __init__(self, name='globalSig',threshold = 0.5,sigSumWeight = None,bkgSumWeight = None, **kwargs):
    super(globalSignificance, self).__init__(name=name, **kwargs)
    self.sigPart = self.add_weight(name='sigPart', initializer='zeros')
    self.bkgPart = self.add_weight(name='bkgPart', initializer='zeros')
    self.sigPartTotal = self.add_weight(name='sigPartTotal', initializer='zeros')
    self.bkgPartTotal = self.add_weight(name='bkgPartTotal', initializer='zeros')
    self.threshold = threshold
    self.sigSumWeight = sigSumWeight.astype('float32')
    self.bkgSumWeight = bkgSumWeight.astype('float32')
	  
  '''
  The next method implements the update of state variables that record training metrics between batches
  '''
  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.cast(y_true, tf.bool) # cast float to boolean array
    y_pred = tf.squeeze(y_pred) # This line corrects a bug I didn't noticed in 2 years: it changes the y_pred size from (batch_size,1) to (batch_size,). That corrected the wrong final result I was obtaining with the metric

    y_pred = tf.cast(tf.greater(y_pred,self.threshold), tf.bool) # Set a value to True whenever the model assigns a value greater than the self.threshold parameter to an element in the prediction array.
    isSig = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True)) # Sets True only to the true positive array elements
    isSig = tf.cast(isSig, self.dtype) # Cast from boolean array to tf.float32 array
    isBkg = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True)) # Sets True only to the false positive array elements
    isBkg = tf.cast(isBkg, self.dtype) # Cast from boolean array to tf.float32 array
    isTotalSig = y_true # Array that is true whenever the batch element comes from a signal event
    isTotalSig = tf.cast(isTotalSig, self.dtype) # cast from boolean array to tf.float32 array
    isTotalBkg = tf.logical_not(y_true) # Array that is true only if the target if from a background event
    isTotalBkg = tf.cast(isTotalBkg, self.dtype) # cast from boolean array to tf.float32 array
  
    if sample_weight is not None:
      sample_weight = tf.cast(sample_weight, self.dtype) # Cast sample_weight to tf.float32
      isSig = tf.multiply(isSig, sample_weight) # Normalized vector of true positives: if the element in the batch is from a true positive then it receives the normalized weight corresponding to it 
      isBkg = tf.multiply(isBkg, sample_weight) # Normalized vector of false positives: if the element is from a false positive then it receives the normalized weight corresponding to it
      isTotalSig = tf.multiply(isTotalSig, sample_weight) # Normalized vector of weights of all the signal events in the batch
      isTotalBkg = tf.multiply(isTotalBkg, sample_weight) # Normalized vector of weights of all the background events in the batch

    self.sigPart.assign_add(self.sigSumWeight*tf.reduce_sum(isSig)) # Set the self.sigPart variable with the physical world number of true positive events in the current batch (have to denormalize to calculate significance)
    self.bkgPart.assign_add(self.bkgSumWeight*tf.reduce_sum(isBkg)) # Set the self.bkgPart variable with the physical world number of false positive events in the current batch (have to denormalize to calculate significance)
    self.sigPartTotal.assign_add(self.sigSumWeight*tf.reduce_sum(isTotalSig)) # the physical world number of signal events in the batch
    self.bkgPartTotal.assign_add(self.bkgSumWeight*tf.reduce_sum(isTotalBkg)) # the physical world number of background events in the batch

  '''
  In the result method of the globalSignificance class I calculate the projected significance. Since I'am using only the 2018 MC it's important to scale it properly to predict how great the significance would be if I was
  using all the run II MC 1.513274 * [sqrt(2018 total number of events) / sqrt(total number of events assessed up to the current batch)] * [self.sigPart / sqrt(self.bkgPart)]. The value 1.513274 is due to the term
  sqrt(137.4/60), which is nothing more than a scale factor to prospect how great the significancy would be if the model was applied to the entire run II.
  '''
  def result(self):
    return 1.513274*tf.math.divide(tf.math.multiply(tf.math.sqrt(self.sigSumWeight+self.bkgSumWeight),self.sigPart),\
                          tf.math.multiply(tf.sqrt(self.bkgPartTotal + self.sigPartTotal),tf.math.sqrt(self.bkgPart)))

  '''
  The next method resets only the relevant state variables—the ones that need to be cleared or nullified at the end of a training epoch.
  '''
  def reset_state(self):
    self.sigPart.assign(0)
    self.bkgPart.assign(0)
    self.sigPartTotal.assign(0)
    self.bkgPartTotal.assign(0)
