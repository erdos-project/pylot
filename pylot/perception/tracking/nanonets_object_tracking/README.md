## Object Tracking
Installation: Use this command to install all the necessary packages. Note that we are using ```python3```

Link to the blog [click here](https://blog.nanonets.com/object-tracking-deepsort/)
```sh
pip install -r requirements.txt
```
This module is built on top of the original deep sort module https://github.com/nwojke/deep_sort
Since, the primary objective is to track objects, We assume that the detections are already available to us, for the given video. The   ``` det/``` folder contains detections from Yolo, SSD and Mask-RCNN for the given video.

```deepsort.py``` is our bridge class that utilizes the original deep sort implementation, with our custom configs. We simply need to specify the encoder (feature extractor) we want to use and pass on the detection outputs to get the tracked bounding boxes. 
```test_on_video.py``` is our example code, that runs deepsort on a video whose detection bounding boxes are already given to us. 

# A simplified overview:
```sh
#Initialize deep sort object.
deepsort = deepsort_rbc(wt_path='ckpts/model640.pt') #path to the feature extractor model.

#Obtain all the detections for the given frame.
detections,out_scores = get_gt(frame,frame_id,gt_dict)

#Pass detections to the deepsort object and obtain the track information.
tracker,detections_class = deepsort.run_deep_sort(frame,out_scores,detections)

#Obtain info from the tracks.
for track in tracker.tracks:
    bbox = track.to_tlbr() #Get the corrected/predicted bounding box
    id_num = str(track.track_id) #Get the ID for the particular track.
    features = track.features #Get the feature vector corresponding to the detection.
```
The ```tracker``` object returned by deepsort contains all necessary info like the track_id, the predicted bounding boxes and the corresponding feature vector of the object. 

Download the test video from [here](https://drive.google.com/open?id=1h2Wnb98tDVB6JlCDNQXCeZpG20x6AiZ2).

The pre-trained weights of the feature extractor are present in ```ckpts/``` folder.
With the video downloaded and all packages installed correctly, you should be able to run the demo with

```sh
python test_on_video.py
```
If you want to train your own feature extractor, proceed to the next section.
# Training a custom feature extractor 
Since, the original deepsort focused on MARS dataset, which is based on people, the feature extractor is trained on humans. We need an equivalent feature extractor for vehicles. We shall be training a Siamese network for the same. More info on siamese nets can be found  [here](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) and [here](https://towardsdatascience.com/lossless-triplet-loss-7e932f990b24)

We have a training and testing set, extracted from the NVIDIA AI city Challenge dataset. You can download it from [here](https://nanonets.s3-us-west-2.amazonaws.com/blogs/object-tracking-crops-data.tar.gz).
 
Extract the ```crops``` and ```crops_test``` folders in the same working directory. Both folders have 184 different sub-folders, each of which contains crops of a certain vehicle, shot in various views. 
Once, the folders have been extracted, we can go through the network configurations and the various options in ```siamese_net.py``` and ```siamese_dataloader.py```. If satisfied, we can start the training process by:
```sh 
python siamese_train.py
```
The trained weights will be stored in ```ckpts/``` folder. We can use ```python siamese_test.py``` to test the accuracy of the trained model. 
Once trained, this model can be plugged in to our deepsort class instance.
