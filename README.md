# Autoencoder Node Saliency (ANS)

The ANS code performs autoencoder node saliency algorithm that helps the researchers in the following
-	To understand what an autoencoder has learned through its unsupervised learning process.  
-	To rank hidden nodes according to their capability of performing a learning task.
-	To identify the specialty nodes that reveal explanatory input features.

This tool is important in transfer learning where the number of samples in a data set is too small for a reasonable learning model to build on. Therefore, features are learned from a much larger collection of data and then transferred to the small data. ANS provides a deeper insight into the features in the reduced dimension.

ANS is also useful when there exist possible faulty class labels on the data. An autoencoder is an unsupervised learning method. Class labels are not used when generating the features in the reduced dimension. Hence, the autoencoder avoids misled by faulty labels in its learning process.

To operate the code, one should have the following Python modules installed:
  - numpy
  - matplotlib

## Example
An example of applying ANS on the MNIST dataset can be found in the Jupyter notebook:

mnist_example.ipynb

## Reference:
Autoencoder Node Saliency: Selecting Relevant Latent Representations
https://doi.org/10.1016/j.patcog.2018.12.015

