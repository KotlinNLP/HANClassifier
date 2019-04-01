# HANClassifier [![GitHub version](https://badge.fury.io/gh/KotlinNLP%2FHANClassifier.svg)](https://badge.fury.io/gh/KotlinNLP%2FHANClassifier) [![Build Status](https://travis-ci.org/KotlinNLP/HANClassifier.svg?branch=master)](https://travis-ci.org/KotlinNLP/HANClassifier)

HANClassifier is a very simple to use text classifier which uses the Hierarchical Attention Networks (HAN) from the [SimpleDNN](https://github.com/KotlinNLP/SimpleDNN "SimpleDNN") library.

HANClassifier is part of [KotlinNLP](http://kotlinnlp.com/ "KotlinNLP").


## Getting Started

### Import with Maven

```xml
<dependency>
    <groupId>com.kotlinnlp</groupId>
    <artifactId>hanclassifier</artifactId>
    <version>0.6.2</version>
</dependency>
```

### Examples

Try some examples of usage of HANClassifier running the files in the `examples` folder.

To run the examples you need datasets of test and training that you can find
[here](https://www.dropbox.com/ "HANClassifier examples datasets")

### Model Serialization

The trained model is all contained into a single class which provides simple dump() and load() methods to serialize it and afterwards load it.


## License

This software is released under the terms of the 
[Mozilla Public License, v. 2.0](https://mozilla.org/MPL/2.0/ "Mozilla Public License, v. 2.0")


## Contributions

We greatly appreciate any bug reports and contributions, which can be made by filing an issue or making a pull 
request through the [github page](https://github.com/KotlinNLP/HANClassifier "HANClassifier on GitHub").
