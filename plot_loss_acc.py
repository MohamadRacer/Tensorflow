import matplotlib.pyplot as plt 
import pandas as pd 

def plot_loss_acc(results) :
  plt.figure(figsize = (15,5))
  plt.subplot(1 , 2 , 1)
  plt.plot(pd.DataFrame(results.history["val_loss"]) , label = "val_loss") ;
  plt.plot(pd.DataFrame(results.history["loss"]) , label = "loss") ;
  plt.legend() ;
  plt.subplot(1 , 2 , 2)
  plt.plot(pd.DataFrame(results.history["accuracy"]) , label = "accuracy")
  plt.plot(pd.DataFrame(results.history["val_accuracy"]) , label = "val_accuracy")
  plt.legend()
