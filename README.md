# NeuronalNetSharp

# Build Status
[![Build status](https://ci.appveyor.com/api/projects/status/fa8xwx98a0n60u3r/branch/development?svg=true)](https://ci.appveyor.com/project/FlorianSestak/neuronalnetsharp/branch/development)

# Instructions
```C#
  var lambda = 0.00;
  var alpha = 0.5;
  var numberOfHiddenLayers = 1;
  var inputLayerSize = 400;
  var outputLayerSize = 10;
  var network = new NeuronalNetwork(inputLayerSize, outputLayerSize, numberOfHiddenLayers, lambda);
```
