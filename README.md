# NeuronalNetSharp
NeuronalNetSharp is a framework for creating, training and testing of neuronal networks. It is a used for my Bachelorwork.

# Build Status
[![Build status](https://ci.appveyor.com/api/projects/status/fa8xwx98a0n60u3r/branch/development?svg=true)](https://ci.appveyor.com/project/FlorianSestak/neuronalnetsharp/branch/development)

# Instructions

```C#
  // Parameter
  var lambda = 0.00;
  var alpha = 0.5;
  var numberOfHiddenLayers = 1;
  var inputLayerSize = 400;
  var outputLayerSize = 10;
  
  // The GetLabelMatrices function looks in the traingdata for all distinct labels and creates 
  var labelMatrices = HelperFunctions.GetLabelMatrices(rawData);
  
  
  // Create network
  var network = new NeuronalNetwork(inputLayerSize, outputLayerSize, numberOfHiddenLayers, lambda);
  
  // Create optimizer
  var optimizer = new GradientDescentAlgorithm(lambda, alpha);
  
  // Optimize network
  optimizer.OptimizeNetwork(network, datas, labelMatrices, 10);
```
