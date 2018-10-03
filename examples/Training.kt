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
 * Train and validate a HAN classifier, using the datasets given as arguments.
 *
 * Command line arguments:
 *   1. The number of classes
 *   2. The name of the file in which to save the model
 *   3. The filename of the training dataset
 *   4. The filename of the validation set
 *   5. The filename of the test set
 *   6. The filename of the pre-trained embeddings
 *   7. The filename of the LHRParser model
 *   8. The filename of the morphology dictionary
 */
fun main(args: Array<String>) {

  val lssModel: LSSModel<ParsingToken, ParsingSentence> = args[6].let {
    println("Loading the LSSEncoder model from the LHRParser model serialized in '$it'...")
    LHRModel.load(FileInputStream(File(it))).lssModel
  }

  val tokensEncoder: TokensEncoder<FormToken, Sentence<FormToken>> = buildTokensEncoder(
    embeddingsMap = args[5].let {
      println("\n-- LOADING EMBEDDINGS FROM '$it'...")
      EMBDLoader().load(it)
    },
    lssModel = lssModel,
    preprocessor = args[7].let {
      println("Loading serialized dictionary from '$it'...")
      MorphoPreprocessor(MorphologicalAnalyzer(
        language = lssModel.language,
        dictionary = MorphologyDictionary.load(FileInputStream(File(it)))))
    })

  println("\n-- READING DATASET:")
  println("\ttraining:   ${args[2]}")
  println("\tvalidation: ${args[3]}")
  println("\ttest:       ${args[4]}")

  val corpusReader = CorpusReader(tokensEncoder)
  val dataset = Dataset(
    training = corpusReader.read(args[2]),
    validation = corpusReader.read(args[3]),
    test = corpusReader.read(args[4]))

  val model = HANClassifierModel(
    tokensEncodingsSize = tokensEncoder.model.tokenEncodingSize,
    attentionSize = 100,
    recurrentConnectionType = LayerType.Connection.RAN,
    outputSize = args[0].toInt())

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
    modelFilename = args[1])

  println("\n-- START VALIDATION ON %d TEST SENTENCES".format(dataset.test.size))

  val accuracy: Double = ValidationHelper(model).validate(testSet = dataset.test)

  println("Accuracy: %.2f%%".format(100.0 * accuracy))
}
