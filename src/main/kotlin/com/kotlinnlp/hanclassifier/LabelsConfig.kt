/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.hanclassifier

import com.beust.klaxon.Json
import com.beust.klaxon.Klaxon
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import java.io.File

/**
 * The configuration of a hierarchy of labels (associated to the classes indices) for the classification.
 *
 * @property labels the list of labels, one for each class at the related index
 * @property subLevels the labels configurations of the sub-levels, one for each class at the related index (null if the
 *                     related class has no sub-levels)
 */
data class LabelsConfig(
  val labels: List<String>,
  @Json(name = "sublevels") val subLevels: List<LabelsConfig?>? = null
) {

  /**
   * Factory object.
   */
  companion object {

    /**
     * Build a labels configuration from a JSON file containing a JSON object with the following template:
     * {
     *  "labels": ["Label 0 Name", "Label 1 Name", "Label 2 Name"]
     *  "sublevels": [{ /* sublevel 0 config */ }, null, { /* sublevel 2 config */ }]
     * }
     *
     * @param filePath the path of the JSON file with a labels configuration
     *
     * @return
     */
    fun fromFile(filePath: String): LabelsConfig = Klaxon().parse(File(filePath))!!
  }

  /**
   * Get the label corresponding to a classes prediction.
   *
   * @param classesPrediction the predictions of a classes hierarchy made by the [HANClassifier]
   *
   * @return the label of to the given prediction
   */
  fun getLabel(classesPrediction: List<DenseNDArray>): String {

    require(classesPrediction.isNotEmpty())

    // Note: only the last can be a 'stop-level prediction.
    val indices: List<Int> = classesPrediction
      .map { it.argMaxIndex() }
      // remove the 'stop-level' predictions (the last index predicted in a sub-level)
      .filterIndexed { level, classIndex -> level == 0 || classIndex != classesPrediction[level].lastIndex }

    var curLevelConfig: LabelsConfig = this

    if (indices.size > 1)
      indices.take(indices.size - 1).forEach { classIndex ->
        curLevelConfig = curLevelConfig.subLevels!![classIndex]!!
      }

    return curLevelConfig.labels[indices.last()]
  }

  /**
   * @param classesConfig a classes configuration
   *
   * @return true if this labels configuration is compatible with the given classes configuration, otherwise false
   */
  fun isCompatible(classesConfig: ClassesConfig): Boolean {

    if (classesConfig.classes.keys.size != this.labels.size) return false

    return if (this.subLevels == null)
      classesConfig.isEmpty
    else
      this.subLevels.withIndex().all { (index, subLevel) ->
        val subClassesConfig: ClassesConfig? = classesConfig.classes[index]
        (subLevel == null && subClassesConfig == null) ||
          (subLevel != null && subClassesConfig != null && subLevel.isCompatible(subClassesConfig))
      }
  }
}
