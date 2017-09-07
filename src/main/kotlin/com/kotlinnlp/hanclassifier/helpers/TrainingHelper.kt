/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.hanclassifier.helpers

import com.kotlinnlp.hanclassifier.HANClassifier
import com.kotlinnlp.hanclassifier.dataset.Example
import com.kotlinnlp.hanclassifier.utils.toHierarchyGroup
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.dataset.Shuffler
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.han.HANParameters
import com.kotlinnlp.simplednn.helpers.training.utils.ExamplesIndices
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.utils.progressindicator.ProgressIndicatorBar
import java.io.File
import java.io.FileOutputStream

/**
 * A helper for the training of a [HANClassifier].
 *
 * @property classifier the [HANClassifier] to train
 * @param updateMethod the [UpdateMethod] for the parameters of the [classifier]
 */
class TrainingHelper(
  private val classifier: HANClassifier,
  updateMethod: UpdateMethod
) {

  /**
   * When timing started.
   */
  private var startTime: Long = 0

  /**
   * The best accuracy reached during the training.
   */
  private var bestAccuracy: Double = 0.0

  /**
   * The helper for the valdiation of the [classifier].
   */
  private val validationHelper = ValidationHelper(this.classifier)

  /**
   * The optimizer of the parameters of the [classifier].
   */
  private val optimizer: ParamsOptimizer<HANParameters> = ParamsOptimizer(
    params = this.classifier.model.params,
    updateMethod = updateMethod)

  /**
   * Train the [classifier] using the given [trainingSet], validating each epoch if a [validationSet] is given.
   *
   * @param trainingSet the dataset to train the classifier
   * @param epochs the number of epochs for the training
   * @param batchSize the size of each batch of examples (default = 1)
   * @param shuffler the [Shuffler] to shuffle the training sentences before each epoch (can be null)
   * @param validationSet the dataset to validate the classifier after each epoch (default = null)
   * @param modelFilename the name of the file in which to save the best trained model (default = null)
   */
  fun train(trainingSet: ArrayList<Example>,
            epochs: Int,
            batchSize: Int = 1,
            shuffler: Shuffler? = Shuffler(enablePseudoRandom = true, seed = 743),
            validationSet: ArrayList<Example>? = null,
            modelFilename: String? = null) {

    (0 until epochs).forEach { i ->

      println("\nEpoch ${i + 1}")

      this.startTiming()

      this.trainEpoch(trainingSet = trainingSet, batchSize = batchSize, shuffler = shuffler)

      println("Elapsed time: %s".format(this.formatElapsedTime()))

      if (validationSet != null) {
        this.validateAndSaveModel(validationSet = validationSet, modelFilename = modelFilename)
      }
    }
  }

  /**
   * Train the HAN classifier on one epoch.
   *
   * @param trainingSet the training set
   * @param batchSize the size of each batch of examples
   * @param shuffler the [Shuffler] to shuffle examples before training (can be null)
   */
  private fun trainEpoch(trainingSet: ArrayList<Example>,
                         batchSize: Int,
                         shuffler: Shuffler?) {

    val progress = ProgressIndicatorBar(trainingSet.size)

    for (exampleIndex in ExamplesIndices(size = trainingSet.size, shuffler = shuffler)) {

      progress.tick()

      this.learnFromExample(example = trainingSet[exampleIndex])

      if ((exampleIndex + 1) % batchSize == 0 || exampleIndex == trainingSet.lastIndex) {
        this.optimizer.update()
      }
    }
  }

  /**
   * Learn from the given [example], comparing its gold output class with the one of the [classifier] and accumulate
   * the propagated errors.
   *
   * @param example the example from which to learn
   */
  private fun learnFromExample(example: Example) {

    val output: DenseNDArray = this.classifier.encoder.forward(
      sequencesHierarchy = example.inputText.toHierarchyGroup(this.classifier.embeddings))

    val errors: DenseNDArray = output.copy()
    errors[example.outputGold] = errors[example.outputGold] - 1

    this.classifier.encoder.backward(outputErrors = errors, propagateToInput = false)

    this.optimizer.accumulate(this.classifier.encoder.getParamsErrors(copy = false))
  }

  /**
   * Validate the [classifier] on the [validationSet] and save its best model to [modelFilename].
   *
   * @param validationSet the validation dataset to validate the [classifier]
   * @param modelFilename the name of the file in which to save the best model of the [classifier] (default = null)
   */
  private fun validateAndSaveModel(validationSet: ArrayList<Example>, modelFilename: String?) {

    val accuracy = this.validateEpoch(validationSet)

    println("Accuracy: %.2f%%".format(100.0 * accuracy))

    if (modelFilename != null && accuracy > this.bestAccuracy) {

      this.bestAccuracy = accuracy

      this.classifier.model.dump(FileOutputStream(File(modelFilename)))

      println("NEW BEST ACCURACY! Model saved to \"$modelFilename\"")
    }
  }

  /**
   * Validate the [classifier] after trained it on an epoch.
   *
   * @param validationSet the validation dataset to validate the [classifier]
   *
   * @return the current accuracy of the [classifier]
   */
  private fun validateEpoch(validationSet: ArrayList<Example>): Double {

    println("Epoch validation on %d sentences".format(validationSet.size))

    return this.validationHelper.validate(validationSet)
  }

  /**
   * Start registering time.
   */
  private fun startTiming() {
    this.startTime = System.currentTimeMillis()
  }

  /**
   * @return the formatted string with elapsed time in seconds and minutes.
   */
  private fun formatElapsedTime(): String {

    val elapsedTime = System.currentTimeMillis() - this.startTime
    val elapsedSecs = elapsedTime / 1000.0

    return "%.3f s (%.1f min)".format(elapsedSecs, elapsedSecs / 60.0)
  }
}
