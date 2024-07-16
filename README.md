# Speaker-Diarization

Speaker Diarization is the procees which aims to find who spoke when in an audio and total number of speakers in an audio recording.
This project contains:

- Voice Activity Detection (webrtcvad)
- Speaker Segmentation based on Bi-LSTM
- Embedding Extraction (d-vector extraction)
- Clustering (k-MEANS and Mean Shift)


## Voice Activity Detection

Voice activity detection (VAD) is a technique in which the presence or absence of human speech is detected. This part has been completed using a module devloped by google called as WebRTC. It's an open framework for the web that enables Real-Time Communications (RTC) capabilities in the browser. The voice activity dector is one of the specific module present in WebRTC. This basic working of WebRTC based VAD is as,

- WebRTC VAD is a Gaussian Mixture Model(GMM) based voice activity detector
- GMM model using PLP features
- Two full covariance Gaussians: One for speech, and one for Non-Speech is used.
  To learn about PLP we followed this paper
  [link](http://www5.informatik.uni-erlangen.de/Forschung/Publikationen/2005/Hoenig05-RPL.pdf).


  ## Speaker Segementation

Speaker segmentation constitues the heart of speaker diarization, the idea to exactly identify the location of speaker change ooint in the order to miliseconds is still an open chalenge. Speaker segmentation constitutes the heart of speaker diarization, the idea to exactly identify the location of speaker change point in the order of milliseconds is still an open challenge. For this part we have tried to develop a state of art system which is BiLSTM network that is trained using a special SMORMS3 optimizer. SMORMS3 optimizer is a hyvrid oprimizer devloped using RMSprop and Adam optimizers. SMORMS3 stands for "squared mean over root mean squared cubed". Following link provied a detailied analysis of SMORMS3 [Link](https://sifter.org/~simon/journal/20150420.html). Now, comming to our speaker segmentation part architecture which ultilizes this SMORMS3 optimizer wroks on the pricipal that address speaker change detection as a binary sequence labeling task using Bidirectional Long Short-Term Memory recurrent neural networks (Bi-LSTM). Given an audio recording, speaker change detection aims at finding the boundaries between speech turns of different speakers. In Figure below, the expected output of such a system would be the list of timestamps between spk1 & spk2, spk2 & spk1, and spk1 & spk4.

- First we extract the features, let x be a sequence of MFCC features extracted on a short (a few milliseconds) overlapping sliding window. The speaker change detection task is then turned into a binary sequence labeling task by defining y = (y1, y2...yT ) ∈ {0, 1}^T
  such that y_{i} = 1 if there is a speaker change during the ith frame, and y_{i} = 0 otherwise. The objective is then to find a function f : X → Y that matches a feature sequence to a label sequence. We propose to model this function f as a recurrent neural network trained using the binary cross-entropy loss function.


  ## Combining VAD and Speaker Segmentation

In our code once the results from above modules were obtained, we combined them the results in logical way such that we had obtained frames of arbitrary seconds depending on the voiced part and the speaker change part. This can be explained with an example, lets us suppose we have first performed VAD and found that from 2 to 3 seconds there is some voice. In next part of speaker segmentation, we found that at 2.5 seconds there is a speaker change point. So, what we did we splited this audio frame of 1 seconds into two parts frame 1 from 2 to 2.5 seconds and then from 2.5 to 3. Similarly lets us suppose that we s=find that from 3 to 3.5 seconds there is some voice and then there is a silence of 1 seconds i.e. exactly at 4 sec some voice is coming into play. Now using Speaker change part we found that at 4 seconds there is speaker changepoint again we combined it in such way we defined there is new speaker at 4 seconds. All such logical results were combined to giver per frame output.


## Embedding Extraction

This part now has to handle the idea to differentiate speakers. As mentioned in previous parts the frames extracted will go through the process of feature extraction. Let’s suppose we have a frame of 3 seconds starting from 4.5 to 7.5 sec, we extract the d-vectors for first 1.5 or 2 seconds of a single frame. To extract d-vectors we use the pyannote libraries pretrained models. The detailed analysis of pyannote can found uisng theor github repo [Link](https://pyannote.github.io/) and also from this [paper](https://arxiv.org/pdf/1911.01255.pdf).

## Clustering (k-MEANS and Mean Shift)

Clustering is one of the most common exploratory data analysis technique used to get an intuition about the structure of the data. It can be defined as the task of identifying subgroups in the data such that data points in the same subgroup (cluster) are very similar while data points in different clusters are very different. In other words, we try to find homogeneous subgroups within the data such that data points in each cluster are as similar as possible according to a similarity measure such as euclidean-based distance or correlation-based distance. The decision of which similarity measure to use is application-specific.

### Kmeans Algorithm

Kmeans algorithm is an iterative algorithm that tries to partition the dataset into Kpre-defined distinct non-overlapping subgroups (clusters) where each data point belongs to only one group. It tries to make the intra-cluster data points as similar as possible while also keeping the clusters as different (far) as possible. It assigns data points to a cluster such that the sum of the squared distance between the data points and the cluster’s centroid (arithmetic mean of all the data points that belong to that cluster) is at the minimum. The less variation we have within clusters, the more homogeneous (similar) the data points are within the same cluster.
The way kmeans algorithm works is as follows:

- Specify number of clusters K.
- Initialize centroids by first shuffling the dataset and then randomly selecting K data points for the centroids without replacement.
- Keep iterating until there is no change to the centroids. i.e assignment of data points to clusters isn’t changing.
- Compute the sum of the squared distance between data points and all centroids.
- Assign each data point to the closest cluster (centroid).
- Compute the centroids for the clusters by taking the average of the all data points that belong to each cluster. 
  The approach kmeans follows to solve the problem is called Expectation-Maximization. Following [Link](https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a) gives more idea.

### Mean Shift Algorithm

Mean Shift is very similar to the K-Means algorithm, except for one very important factor, you do not need to specify the number of groups prior to training. The Mean Shift algorithm finds clusters on its own. For this reason, it is even more of an "unsupervised" machine learning algorithm than K-Means. Mean shift builds upon the concept of kernel density estimation (KDE). KDE is a method to estimate the underlying distribution for a set of data. It works by placing a kernel on each point in the data set. A kernel is a fancy mathematical word for a weighting function. There are many different types of kernels, but the most popular one is the Gaussian kernel. Adding all of the individual kernels up generates a probability surface (e.g., density function). Depending on the kernel bandwidth parameter used, the resultant density function will vary. 
Mean shift exploits KDE idea by imagining what the points would do if they all climbed up hill to the nearest peak on the KDE surface. It does so by iteratively shifting each point uphill until it reaches a peak. Depending on the kernel bandwidth used, the KDE surface (and end clustering) will be different. As an extreme case, imagine that we use extremely tall skinny kernels (e.g., a small kernel bandwidth). The resultant KDE surface will have a peak for each point. This will result in each point being placed into its own cluster. On the other hand, imagine that we use an extremely short fat kernels (e.g., a large kernel bandwidth). This will result in a wide smooth KDE surface with one peak that all of the points will climb up to, resulting in one cluster. Kernels in between these two extremes will result in nicer clusterings. Below are two animations of mean shift running for different kernel bandwidth values. 
 Following [Link](https://spin.atomicobject.com/2015/05/26/mean-shift-clustering/) gives more idea.

# Dataset

1.Hindi A data is taken from Hindi News Channel Debate from Youtbue Video https://www.youtube.com/watch?v=1Yj8K2ZHttA&t=424s. The duration of dataset is approx 2 Hours. This data set is split into 3 files Hindi_01, Hindi_02 and Hindi_03 having approximately equal duration. . The complete dataset is manually annotated.The annotations are in the format (filename/duration/offset/speaker_id).
2. All the hindi dataset was taken from Youtube Video recording. The audio files (.wav) from Youtube Video were extracted using the (MiniTool uTube Downloader)  and then this files were converted from stereo type to mono type using Audacity software. The spliiting of the files was also done using Audacity. Then splitted files were then exported as .wav files having sampling rate 48000Hz and were 16 bit PCM encoded.
3. Hindi A and Hindi B dataset does not have same speakers. In both the data Speakers are different. None of the Speaker is same.
\
