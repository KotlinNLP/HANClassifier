/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import com.kotlinnlp.hanclassifier.HANClassifier
import com.kotlinnlp.hanclassifier.HANClassifierModel
import com.kotlinnlp.hanclassifier.dataset.CorpusReader
import com.kotlinnlp.hanclassifier.dataset.Dataset
import com.kotlinnlp.hanclassifier.helpers.Trainer
import com.kotlinnlp.hanclassifier.helpers.Validator
import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.tokensencoder.embeddings.keyextractor.NormWordKeyExtractor
import com.kotlinnlp.simplednn.core.embeddings.EMBDLoader
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMapByDictionary
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adagrad.AdaGradMethod
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.tokensencoder.embeddings.EmbeddingsEncoderModel
import com.kotlinnlp.tokensencoder.reduction.ReductionEncoder
import com.kotlinnlp.tokensencoder.reduction.ReductionEncoderModel
import com.kotlinnlp.utils.stats.MetricCounter
import com.xenomachina.argparser.mainBody

/**
 * Train and validate a [HANClassifierModel].
 *
 * Launch with the '-h' option for help about the command line arguments.
 */
fun main(args: Array<String>) = mainBody {

  val parsedArgs = CommandLineArguments(args)

  val embeddingsMap: EmbeddingsMapByDictionary = parsedArgs.embeddingsPath.let {
    println("Loading embeddings from '$it'...")
    EMBDLoader().load(it)
  }

  val embeddingsEncoderModel = EmbeddingsEncoderModel<FormToken, Sentence<FormToken>>(
    embeddingsMap = embeddingsMap,
    embeddingKeyExtractor = NormWordKeyExtractor())

  val tokensEncoder = ReductionEncoder(
    model = ReductionEncoderModel(
      inputEncoderModel = embeddingsEncoderModel,
      tokenEncodingSize = 50,
      activationFunction = Tanh()),
    useDropout = true
  )

  val corpusReader = CorpusReader()
  val dataset = Dataset(
    training = parsedArgs.trainingSetPath.let {
      println("Loading training dataset from '$it'...")
      corpusReader.read(it)
    },
    validation = parsedArgs.validationSetPath.let {
      println("Loading validation dataset from '$it'...")
      corpusReader.read(it)
    },
    test = parsedArgs.testSetPath.let {
      println("Loading test dataset from '$it'...")
      corpusReader.read(it)
    })

  dataset.training.forEach { example ->
    example.sentences.forEach { s -> s.tokens.forEach { embeddingsEncoderModel.embeddingsMap.dictionary.add(it.form) } }
  }

  val model = HANClassifierModel(
    name = parsedArgs.modelName,
    classesConfig = dataset.classesConfig,
    tokensEncodingsSize = tokensEncoder.model.tokenEncodingSize,
    attentionSize = 100,
    recurrentConnectionType = LayerType.Connection.RAN)

  println("\n-- START TRAINING ON %d SENTENCES".format(dataset.training.size))

  Trainer(
    classifier = HANClassifier(
      model = model,
      useDropout = true,
      propagateToInput = true),
    updateMethod = ADAMMethod(stepSize = 0.001),
    tokensEncoder = tokensEncoder,
    tokensEncoderOptimizer = tokensEncoder.model.buildOptimizer(updateMethod = AdaGradMethod(learningRate = 0.1)),
    onSaveModel = {
      println("Saving re-trained embeddings to '${parsedArgs.trainedEmbeddingsPath}'...")
      embeddingsMap.dump(parsedArgs.trainedEmbeddingsPath, digits = 6)
    }
  ).train(
    trainingSet = dataset.training,
    validationSet = dataset.validation,
    epochs = 10,
    modelFilename = parsedArgs.modelPath)

  println("\n-- START VALIDATION ON %d TEST SENTENCES".format(dataset.test.size))

  val metrics: List<MetricCounter> =
    Validator(model = model, tokensEncoderModel = tokensEncoder.model).validate(testSet = dataset.test)
  val accuracy: Double = metrics.map { it.f1Score }.average()

  println("Final accuracy (f1 average): %5.2f %%".format(100 * accuracy))
  metrics.forEachIndexed { i, it -> println("- Level $i: $it") }
}
