'''The next code cell is the main code cell'''

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from preprocessMC import dataset, dataset_W, dataset_Y, sigSumWeight, bkgSumWeight
from significanceMetric import globalSignificance
from bestModelCallback import CustomModelCheckpoint

valTestSplit = 0.25 # 25% of the shuffled 2-D numpy array that represents our dataset will be assigned to as the validation split
epochs=200 # Number of training epochs

stopTraining = False # This variable is set to True every time the CustomModelCheckpoint class finds a model that fulfills all the preset conditions that a chosen model must reach. 
while (not stopTraining):
    ''' 
    The next line randomly splits the dataset into training and validation sets.
    '''
    dataset_train, dataset_test, dataset_train_W, dataset_test_W, dataset_train_Y, dataset_test_Y\
                                            = train_test_split(dataset,dataset_W,dataset_Y\
                                            ,test_size=valTestSplit,shuffle=True)
    '''
    The next 4 lines define the used machine learning architecture: shallow MLP with 200 neurons in the hidden layer, one input layer and one output layer with a single neuron 
    '''
    model = models.Sequential()
    model.add(layers.Input(shape=(21,)))
    model.add(layers.Dense(200, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    optim = Adam(learning_rate=0.001) # The optimizer choice
    globalSignif = globalSignificance(threshold=0.94,sigSumWeight=sigSumWeight.astype('float32'),bkgSumWeight=bkgSumWeight.astype(dtype='float32')) # Instantiation of the "globalSignificance" custom metric object and the threshold

    '''
    The next line compiles the model, sets weighted metric and optimizer. "run_eagerly" was/is only used for debugging purposes.
    '''
    model.compile(optimizer=optim, loss='binary_crossentropy', weighted_metrics=[globalSignif],run_eagerly=False)

    '''
    The next line instantiates a CustomModelCheckpoint object to be passed to the model.fit method. As one can see in the next line, we set the preset requirements such as "minimum_acceptable_val_metric", which is the minimum
    acceptable validation metric that a model must have to be chosen, the "trainValidation_percent_threshold" of 1%, which is the largest percentual deviation the validation metric value must have w.r.t the training metric
    value, for a model to be chosen, the monitor is the "globalSig" custom metric and the mode is max, pointing out that we are searchig for the highest possible metric values. The number of training epochs is also important
    internally for the class, the name of the custom metric class object and "filepath" are also important
    '''
    modelCheckPoint = CustomModelCheckpoint(globalSignif,filepath,monitor='globalSig',mode='max',save_best_only=True,trainValidation_percent_threshold = 1.0,trainingEpochs = epochs\
                                            ,minimum_accepted_val_metric = 1.9)

    '''
    This next line sets the batch size, which I've prefired to set as a function of the number of batches that a dataset provides us. I've always procceed to prime factorize the "len(dataset_train)" quantity and try to set
    the number of batches as the integer divisor of "len(dataset_train)" that lies as close as possible to 1000. This is due to the appearance of negative loss values, in some training epochs, when I use a very high number
    of batches. The other issue that prevents me from using a batch size much larger than 1000 is the possibility of having bins with a negative number of events in the final histogram of the scores: this effect is due to
    the presence of events with negative weights in the 2018 MC. At the same time if I use a number of batches much smaller than 1000 I suffer from the problem of impairing the generalization capabilities of the model as
    well as taking the risk of converging to a local minimum in the optimization landscape.
    '''
    miniBatch = int(round(len(dataset_train)/2000))  
    

    '''
    The next line begins effectively stars the training and validation process, calling the CustomModelCheckpoint callback function and globalSignificance custom metric
    '''
    history = model.fit( dataset_train , dataset_train_Y,
                    epochs=epochs,
                    batch_size=miniBatch,
                    verbose=1,
                    validation_data=(dataset_test,dataset_test_Y,dataset_test_W),sample_weight = dataset_train_W,callbacks= [lr_scheduler,modelCheckPoint])
    
    stopTraining = model.stop_training # This line sets the stopTraining boolean variable to the model.stop_training that was set by the CustomModelCheckpoint callback class

'''
In the next two lines, I load the best model saved by the CustomModelCheckpoint callback class, allowing me to experiment with it and plot the output scores, ROC-AUC curve, or any other metrics I wish to explore.
'''
<chosenFileName = filepath+'best2.keras'
model.load_weights(chosenFileName)
'''------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''
'''
In the next two lines I bring the weights of the events back to the physical world weights, denormalizing signal and background weights
'''
dataset_test_W[dataset_test_Y == 1.0] = dataset_test_W[dataset_test_Y == 1.0]*(sigSumWeight)
dataset_test_W[dataset_test_Y == 0.0] = dataset_test_W[dataset_test_Y == 0.0]*(bkgSumWeight)

datasetcp_W = dataset_W.copy()
datasetcp = dataset.copy()
datasetcp_Y = dataset_Y.copy()
'''
In the next two lines I bring the weights of the entire dataset back to the physical world weights (not only the test sample but everything)
'''
datasetcp_W[datasetcp_Y == 1.0] = datasetcp_W[datasetcp_Y == 1.0]*sigSumWeight
datasetcp_W[datasetcp_Y == 0.0] = datasetcp_W[datasetcp_Y == 0.0]*bkgSumWeight
'''
In the next two lines I obtain the predictions array for the test dataset and for the entire dataset
'''
predictions = np.squeeze(model.predict(dataset_test))
predictions_all = np.squeeze(model.predict(dataset))
'''-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''
'''
In the next lines we plot the loss and metric with the epoch index number in the x axis and those informed quantities in the y
'''
from sklearn.metrics import accuracy_score

fig_hist = plt.figure(figsize=(22,8))
ax_fig = fig_hist.add_subplot(1,2,1)

# Plot loss vs epoch
ax_fig.plot(history.history['loss'], label='loss')
ax_fig.plot(history.history['val_loss'], label='val loss')
ax_fig.legend(loc="upper right",fontsize=16)
ax_fig.set_xlabel('epoch',fontsize=17)
ax_fig.set_ylabel('loss',fontsize=17)
ax_fig.tick_params(axis='both',which='major',labelsize=17)
ax_fig.set_title('Loss of MLP',fontsize=18)
plt.grid()

# Plot accuracy vs epoch
ax_fig = fig_hist.add_subplot(1,2,2)
ax_fig.plot(history.history['globalSig'], label='Significance')
ax_fig.plot(history.history['val_globalSig'], label='Val significance')
ax_fig.legend(loc="lower right",fontsize=16)
ax_fig.set_xlabel('epoch',fontsize=17)
ax_fig.set_ylabel('Significance',fontsize=17)
ax_fig.tick_params(axis='both',which='major',labelsize=17)
ax_fig.set_title('Global significance custom metric plot',fontsize=18)

plt.grid()
'''------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''
'''
This cell is in charge of plotting the output scores histogram and the ROC curve
'''

from sklearn.metrics import roc_curve, auc, accuracy_score
'''
The next 5 lines computes the ROC curve values and the AUC_ROC value 
'''
plt.figure(figsize=(20,7))
fpr, tpr, thresholds = roc_curve(dataset_test_Y, predictions,sample_weight=dataset_test_W)
auc = 0.0
for i in range(len(fpr)-1):
  auc+=((fpr[i+1] - fpr[i]) * tpr[i])
	
ax = plt.subplot(1, 2, 1)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.plot(fpr, tpr, lw=2, color='cyan')
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='random chance')
ax.set_xlim([0, 1.0])
ax.set_ylim([0, 1.0])
ax.set_xlabel('False Positive Rate(FPR)',fontsize=18)
ax.set_ylabel('True Positive Rate(TPR)',fontsize=18)
ax.set_title('Unified MLP\'s ROC curve 60.0 $fb^{-1}$, Run II (13Tev)',fontsize=17)
ax.legend(['auc = %.3f' % (auc)],loc="lower right",fontsize=16)

'''
The next line prints in the screen the Accuracy when we use the threshold of 0.5
'''
print("Accuracy: ",accuracy_score(dataset_test_Y,np.where(predictions>0.5,1,0),sample_weight=dataset_test_W[:]))

# Plot DNN output
ax = plt.subplot(1, 2, 2)
X = np.linspace(0.0, 1.0, 100)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.hist(predictions[dataset_test_Y[:]==1.0], bins=X, label='sig',histtype='step',weights=1e2*dataset_test_W[dataset_test_Y[:]==1.0])
ax.hist(predictions[(dataset_test_Y[:]==0.0)], bins=X, label='bkg',histtype='step',weights=dataset_test_W[(dataset_test_Y[:]==0.0)])
ax.set_xlabel('Output Score',fontsize=18)
ax.set_ylabel('Events Number',fontsize=18)
ax.set_title('Unified MLP\'s output scores 60 $fb^{-1}$, Run II (13Tev)',fontsize=17)
ax.legend(['1e2 *sig','bkg'],loc='upper center',fontsize=16)
plt.grid()
