# FaceID Data Generation

This folder contains scripts to generate data to train and test models for FaceID model using the following datasets:
 - VGGFace-2: A large-scale face recognition dataset. [https://www.robots.ox.ac.uk/~vgg/data/](https://www.robots.ox.ac.uk/~vgg/data/)
 - YouTubeFaces: A database of face videos designed for studying the problem of unconstrained face recognition. [https://www.cs.tau.ac.il/~wolf/ytfaces/](https://www.cs.tau.ac.il/~wolf/ytfaces/)

## Dataset Generation

### VGGFace-2 
**Warning:**  The original dataset is about 40GB and the following scripts generate a new dataset with size of 183 GB. Be sure there is enough space on your hard drive before starting the execution.

\
Follow these steps for both train and test sets.
1. Download train and test the *VGG Face 2 Dataset* from [https://www.robots.ox.ac.uk/~vgg/data/](https://www.robots.ox.ac.uk/~vgg/data/) and extract the .tar.gz files. into the same folder.
2. Run gen_vggface2_embeddings.py:
   ```
   python gen_vggface2_embeddings.py -r <path_to_vggface2_dataset_folder> -d <path_to_store_generated_dataset> --type <train/test>
   ```
3. Run merge_vggface2_dataset.py
   ```
   python merge_vggface2_dataset.py -p <path_to_store_generated_dataset> --type <train/test>
   ```
   
### YouTubeFaces

**Warning:**  The original dataset is about 29GB and the following scripts generate a new dataset with size of 15 GB. Be sure there is enough space on your hard drive before starting the execution.

\
Follow these steps.
1. Download the dataset from [here](http://www.cslab.openu.ac.il/download/) and extract the tar.gz files. into the same folder.
2. Run gen_youtubefaces_embeddings.py:
   ```
   python gen_youtubefaces_embeddings.py -r <path_to_youtubefaces_dataset_folder> -d <path_to_store_generated_dataset> --type test
   ```
3. Run merge_youtubefaces_dataset.py 
   ```
   python merge_youtubefaces_dataset.py -p <path_to_store_generated_dataset> --type test
   ```

**Note:** The default paths for generated dataset is set to AI8X_TRAINING_HOME/data so the data loaders can load them with default parameters. If the destination folder is changed, the
--data <folder_to_generated_dataset> option should be added to the model training script. 
