This project is linked to my master thesis about "__Machine learning based fast jet finding for FPGA real-time processing at the LHC ATLAS experiment__".

It aims to developp a basic machine learning model based on "CNN" that can be able to detect "jets" on images of size 64x64 using as backbone "ResNet18" and adding to it a "head" realising the detection.

An alternative version quantized version with "QKeras" is also developped.

Finally the model is accelerated with "hls4ml" to get a FPGA implementation.

There is also different scripts to manipulate the files containing the data and generate a file with corresponding images (representation of the energy and the pt of the calorimeter cells of a granularity of 0.1x0.1) and labels.
