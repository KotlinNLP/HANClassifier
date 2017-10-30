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
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod

/**
 * Train and validate a HAN classifier, using the datasets given as arguments.
 * The first argument is the file name of the training dataset, the second one of the validation set and the third one
 * of the test set.
 */
fun main(args: Array<String>) {

  println("Start 'HAN Classifier Test'")

  println("\n-- READING DATASET:")
  println("\ttraining:   ${args[0]}")
  println("\tvalidation: ${args[1]}")
  println("\ttest:       ${args[2]}")

  val dataset = Dataset(
    training = CorpusReader.read(args[0]),
    validation = CorpusReader.read(args[1]),
    test = CorpusReader.read(args[2]))

  val classifier = HANClassifier(model = HANClassifierModel(outputSize = 5))

  println("\n-- START TRAINING ON %d SENTENCES".format(dataset.training.size))

  TrainingHelper(classifier = classifier, updateMethod = ADAMMethod(stepSize = 0.0001))
    .train(trainingSet = dataset.training, validationSet = dataset.validation, epochs = 10)

  println("\n-- START VALIDATION ON %d TEST SENTENCES".format(dataset.test.size))

  val accuracy: Double = ValidationHelper(classifier).validate(testSet = dataset.test)

  println("Accuracy: %.2f%%".format(100.0 * accuracy))
}
