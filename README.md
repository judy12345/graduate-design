# Sparse-Adversarial-Attack
Implementation for **Bechlor graduate design**

Deep learning general object detectors have been widely deployed in practical applications such as face detection,autonomous driving, and medical image detection after extensive research and development. However, deep neural networks are easily disturbed by adversarial attacks due to their perceived vulnerability to adversarial samples.Therefore, designing and researching adversarial samples can help to evaluate the reliability of deep object detectors and ameliorate their deficiencies.
In this work, we propose an adversarial sample attack method namely APA: an untargeted attack method based on attention mechanism. The detection and classification performance of the target detector is attacked by adding "cross-
shaped" patches to the key parts of the image (i.e., the feature area of the input image that the target detector pays attention to and the horizontal and vertical areas where the center point of the detection frame is located).

To reproduce the attack performance:
   Run ```improve_1.py``` to generate the adversarial images
