/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package training.classifiersonly

import com.kotlinnlp.hanclassifier.HANClassifierModel
import com.kotlinnlp.hanclassifier.MultiLevelHANModel
import com.kotlinnlp.hanclassifier.dataset.CorpusReader
import com.kotlinnlp.hanclassifier.dataset.Dataset
import com.kotlinnlp.hanclassifier.helpers.Trainer
import com.kotlinnlp.hanclassifier.helpers.Validator
import com.kotlinnlp.tokensencoder.embeddings.keyextractor.NormWordKeyExtractor
import com.kotlinnlp.simplednn.core.embeddings.EMBDLoader
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMapByDictionary
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.tokensencoder.embeddings.EmbeddingsEncoderModel
import com.xenomachina.argparser.mainBody
import training.CommandLineArguments
import java.io.File
import java.io.FileInputStream

/**
 * Train and validate a [HANClassifierModel] without optimizing the tokens encoder.
 * Only the [MultiLevelHANModel] will be serialized.
 *
 * Launch with the '-h' option for help about the command line arguments.
 */
fun main(args: Array<String>) = mainBody {

  val parsedArgs = CommandLineArguments(args)

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

  val embeddingsMap: EmbeddingsMapByDictionary = parsedArgs.embeddingsPath.let {
    println("Loading embeddings from '$it'...")
    EMBDLoader().load(it)
  }

  dataset.training.forEach { example ->
    example.sentences.forEach { s -> s.tokens.forEach { embeddingsMap.dictionary.add(it.form) } }
  }

  val tokensEncoderModel = EmbeddingsEncoderModel.Transient(
    embeddingsMap = embeddingsMap,
    embeddingKeyExtractor = NormWordKeyExtractor())

  val model = HANClassifierModel(
    name = parsedArgs.modelName,
    classesConfig = dataset.classesConfig,
    tokensEncoder = tokensEncoderModel,
    attentionSize = 100,
    recurrentConnectionType = LayerType.Connection.RAN)

  println("\n-- START TRAINING ON %d SENTENCES".format(dataset.training.size))

  Trainer(
    model = model,
    classifierUpdateMethod = ADAMMethod(stepSize = 0.001),
    tokensEncoderUpdateMethod = null,
    useDropout = true,
    saveClassifiersOnly = true
  ).train(
    trainingSet = dataset.training,
    validationSet = dataset.validation,
    epochs = 10,
    modelFilename = parsedArgs.modelPath)

  println("\n-- START VALIDATION ON %d TEST SENTENCES".format(dataset.test.size))

  val validationModel = HANClassifierModel(
    multiLevelHAN = MultiLevelHANModel.load(FileInputStream(File(parsedArgs.modelPath))), // load the best model
    tokensEncoder = tokensEncoderModel
  )
  val info: Validator.ValidationInfo = Validator(validationModel).validate(testSet = dataset.test)
  val accuracy: Double = info.metrics.map { it.f1Score }.average()

  println("Final accuracy (f1 average): %5.2f %%".format(100 * accuracy))
  info.metrics.forEachIndexed { i, it -> println("- Level $i: $it") }

  println("Level 0 confusion:")
  println(info.confusionMatrix)
}
