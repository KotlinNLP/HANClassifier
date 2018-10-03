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
import com.kotlinnlp.hanclassifier.helpers.TrainingHelper
import com.kotlinnlp.hanclassifier.helpers.ValidationHelper
import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.lssencoder.LSSModel
import com.kotlinnlp.morphologicalanalyzer.MorphologicalAnalyzer
import com.kotlinnlp.morphologicalanalyzer.dictionary.MorphologyDictionary
import com.kotlinnlp.neuralparser.helpers.preprocessors.MorphoPreprocessor
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.language.ParsingToken
import com.kotlinnlp.neuralparser.parsers.lhrparser.LHRModel
import com.kotlinnlp.simplednn.core.embeddings.EMBDLoader
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.tokensencoder.TokensEncoder
import java.io.File
import java.io.FileInputStream

/**
 * Train and validate a [HANClassifierModel].
 *
 * Launch with the '-h' option for help about the command line arguments.
 */
fun main(args: Array<String>) {

  val parsedArgs = CommandLineArguments(args)

  val lssModel: LSSModel<ParsingToken, ParsingSentence> = parsedArgs.parserModelPath.let {
    println("Loading the LSSEncoder model from the LHRParser model serialized in '$it'...")
    LHRModel.load(FileInputStream(File(it))).lssModel
  }

  val tokensEncoder: TokensEncoder<FormToken, Sentence<FormToken>> = buildTokensEncoder(
    embeddingsMap = parsedArgs.embeddingsPath.let {
      println("\n-- LOADING EMBEDDINGS FROM '$it'...")
      EMBDLoader().load(it)
    },
    lssModel = lssModel,
    preprocessor = parsedArgs.morphoDictionaryPath.let {
      println("Loading serialized dictionary from '$it'...")
      MorphoPreprocessor(MorphologicalAnalyzer(
        language = lssModel.language,
        dictionary = MorphologyDictionary.load(FileInputStream(File(it)))))
    })

  println("\n-- READING DATASET:")
  println("\ttraining:   ${parsedArgs.trainingSetPath}")
  println("\tvalidation: ${parsedArgs.validationSetPath}")
  println("\ttest:       ${parsedArgs.testSetPath}")

  val corpusReader = CorpusReader(tokensEncoder)
  val dataset = Dataset(
    training = corpusReader.read(parsedArgs.trainingSetPath),
    validation = corpusReader.read(parsedArgs.validationSetPath),
    test = corpusReader.read(parsedArgs.testSetPath))

  val model = HANClassifierModel(
    name = parsedArgs.modelName,
    tokensEncodingsSize = tokensEncoder.model.tokenEncodingSize,
    attentionSize = 100,
    recurrentConnectionType = LayerType.Connection.RAN,
    outputSize = parsedArgs.classesNumber)

  println("\n-- START TRAINING ON %d SENTENCES".format(dataset.training.size))

  TrainingHelper(
    classifier = HANClassifier(
      model = model,
      useDropout = true,
      propagateToInput = true),
    updateMethod = ADAMMethod(stepSize = 0.001)
  ).train(
    trainingSet = dataset.training,
    validationSet = dataset.validation,
    epochs = 10,
    modelFilename = parsedArgs.modelPath)

  println("\n-- START VALIDATION ON %d TEST SENTENCES".format(dataset.test.size))

  val accuracy: Double = ValidationHelper(model).validate(testSet = dataset.test)

  println("Accuracy: %.2f%%".format(100.0 * accuracy))
}
