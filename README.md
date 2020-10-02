# EmotionNet2
Building on EmotionNet we improve generalization by improving the way we handle our input data.

## What is EmotionNet?
EmotionNet is an application of Convolutional Neural Networks to emotion recognition in photographs of human faces. EmotionNet2 does face detection in addition to classification to assist in reducing variance due to backgroud noise. See the original [EmotionNet](https://github.com/co60ca/EmotionNet). There is no web interface for this version as was present in the original version.

# Support
You should please note that no support is guaranteed and only best effort is provided. I will gladly accept reasonable patches and would like to hear if documentation/instructions are unclear. By no means are the instructions 100% foolproof but I expect you to be able to have some understanding of Python, and Linux environments.

# Instructions
1. Install requirements, including `requirements.txt`. Use `pip3 --requirement requirements.txt`. Additionally, you need pytorch gpu. 

Follow either Pretrained or Train Yourself streams after completing the requirements.
## Pretrained
2. Download the external checkpoint from [my server](http://data.co60.ca/checkpoint.pth.tar.gz) and place in the root directory. 
3. Then, decompress using gzip, `gunzip checkpoint.pth.tar.gz` (removes .gz extension)
4. Start python3 from the root directory with the `PYTHONPATH=python` as the python scripts are in the python directory.
5. To try one image you can do the following:
```python
import emotionnet
net = emotionnet.EmotionNet(layers=[3, 4, 23, 3])
net.load_checkpoint('checkpoint.pth.tar')
net.classify_one_image('filename of image.jpg')
```
To run this as a service you would want to keep the checkpoint loaded in memory as it takes some time to load.

## Train Yourself
2. In order to train you will need [KDEF](http://kdef.se/) the scripts expect the file References.txt in the KDEF distribution to be in KDEF/References.txt from the root of the project.

3. Once you have KDEF data you need to split the data up into training/validation/testing and extract the faces from the KDEF data.
4. Run `PYTHONPATH=python python3 python/extract_faces.py` which extracts the faces into `train/`
5. Run `PYTHONPATH=python python3 python/split_dataset.py` which splits into train/test/valid
6. Finally run a trainer. This will take some time. It is recommended to make a file containing the below called train.sh and run this file. Set tname on the first line to whatever you like. The tname will be your `checkpoint.pth.tar` from the pretrained.

```bash
tname=model-01
PYTHONPATH=$PYTHONPATH python3 <<HEAD
import emotionnet
net = emotionnet.EmotionNet(layers=[3, 4, 23, 3])
net.train_model('train', 'valid', '$tname', epochs=200, csvout='${tname}-csv.csv')
net.load_checkpoint('best-${tname}.pth.tar')
net.valid_model('valid')
HEAD
```
7. Now you've trained your model, follow the pretrained instructions from step 4. using your checkpoint file and not the `checkpoint.pth.tar`. Your best model will start with `best-`

