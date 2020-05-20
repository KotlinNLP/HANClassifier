/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package training

import com.kotlinnlp.hanclassifier.HANClassifierModel
import com.kotlinnlp.hanclassifier.dataset.CorpusReader
import com.kotlinnlp.hanclassifier.dataset.Dataset
import com.kotlinnlp.hanclassifier.helpers.Trainer
import com.kotlinnlp.hanclassifier.helpers.Validator
import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.tokensencoder.embeddings.keyextractor.NormWordKeyExtractor
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMap
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adagrad.AdaGradMethod
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.radam.RADAMMethod
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.tokensencoder.embeddings.EmbeddingsEncoderModel
import com.kotlinnlp.tokensencoder.reduction.ReductionEncoderModel
import com.xenomachina.argparser.mainBody
import java.io.File
import java.io.FileInputStream

/**
 * Train and validate a [HANClassifierModel].
 *
 * Launch with the '-h' option for help about the command line arguments.
 */
fun main(args: Array<String>) = mainBody {

  val parsedArgs = CommandLineArguments(args)
  val optimizeEmbeddings: Boolean = !parsedArgs.noEmbeddingsOptimization

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
    },
    autoComplete = parsedArgs.autoComplete)

  val embeddingsMap: EmbeddingsMap<String> = parsedArgs.embeddingsPath.let {
    println("Loading pre-trained word embeddings from '$it'...")
    EmbeddingsMap.load(it)
  }

  if (optimizeEmbeddings) embeddingsMap.addAll(dataset.training.flatMap { it.sentences } )

  val tokensEncoderModel = ReductionEncoderModel(
    inputEncoderModel = if (parsedArgs.noEmbeddingsOptimization)
      EmbeddingsEncoderModel.Transient(embeddingsMap = embeddingsMap, embeddingKeyExtractor = NormWordKeyExtractor())
    else
      EmbeddingsEncoderModel.Base(embeddingsMap = embeddingsMap, embeddingKeyExtractor = NormWordKeyExtractor()),
    optimizeInput = optimizeEmbeddings,
    tokenEncodingSize = 50,
    activationFunction = Tanh)

  val model = HANClassifierModel(
    name = parsedArgs.modelName,
    classesConfig = dataset.classesConfig,
    tokensEncoder = tokensEncoderModel,
    attentionSize = 100,
    recurrentConnectionType = LayerType.Connection.RAN)

  println("\n-- START TRAINING ON %d SENTENCES".format(dataset.training.size))

  Trainer(
    model = model,
    classifierUpdateMethod = RADAMMethod(stepSize = 0.001),
    tokensEncoderUpdateMethod = AdaGradMethod(learningRate = 0.1),
    useDropout = true
  ).train(
    trainingSet = dataset.training,
    validationSet = dataset.validation,
    epochs = parsedArgs.epochs,
    modelFilename = parsedArgs.modelPath)

  println("\n-- START VALIDATION ON %d TEST SENTENCES".format(dataset.test.size))

  // Load the best model.
  val validationModel = HANClassifierModel.load(FileInputStream(File(parsedArgs.modelPath)))

  if (parsedArgs.noEmbeddingsOptimization) {

    val inputTokensEncoder: EmbeddingsEncoderModel.Transient<*, *> =
      (validationModel.tokensEncoder as ReductionEncoderModel).inputEncoderModel as EmbeddingsEncoderModel.Transient

    inputTokensEncoder.setEmbeddingsMap(embeddingsMap)
  }

  val info: Validator.ValidationInfo = Validator(validationModel).validate(testSet = dataset.test)
  val accuracy: Double = info.metrics.map { it.f1Score }.average()

  println("Final accuracy (f1 average): %5.2f %%".format(100 * accuracy))
  info.metrics.forEachIndexed { i, it -> println("- Level $i: $it") }

  println("Level 0 confusion:")
  println(info.confusionMatrix)
}

/**
 * Add the all the tokens form to the embeddings map.
 *
 * @param sentences the list of sentences
 */
private fun EmbeddingsMap<String>.addAll(sentences: List<Sentence<FormToken>>) {

  sentences.forEach { sentence ->
    sentence.tokens.forEach { token ->
      if (token.normalizedForm !in this)
        this.set(token.normalizedForm) // random initialized
    }
  }
}
