/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.hanclassifier

import com.beust.klaxon.JsonObject
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The configuration of a hierarchy of labels (associated to the classes indices) for the classification.
 *
 * @property labels the list of labels, one for each class at the related index
 * @property subLevels the labels configurations of the sub-levels, one for each class at the related index (null if the
 *                     related class has no sub-levels)
 */
data class LabelsConfig(val labels: List<String>, val subLevels: List<LabelsConfig?>) {

  /**
   * Factory object.
   */
  companion object {

    /**
     * Build a labels configuration from a JSON object with the following template:
     * {
     *  "labels": ["Label 0 Name", "Label 1 Name", "Label 2 Name"]
     *  "sublevels": [{ /* sublevel 0 config */ }, null, { /* sublevel 2 config */ }]
     * }
     *
     * @param jsonObject a JSON object representing a labels configuration
     *
     * @return
     */
    fun fromJSON(jsonObject: JsonObject): LabelsConfig = LabelsConfig(
      labels = jsonObject.array("labels")!!,
      subLevels = jsonObject.array<JsonObject?>("sublevels")!!.map { obj -> obj?.let { LabelsConfig.fromJSON(it) } }
    )
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

    var curLevelConfig: LabelsConfig = this
    val indices: Sequence<Int> = classesPrediction.asSequence().map { it.argMaxIndex() }

    indices.forEach { classIndex ->
      curLevelConfig = curLevelConfig.subLevels.getOrNull(classIndex) ?: curLevelConfig
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

    return this.subLevels.withIndex().all { (index, subLevel) ->
      val subClassesConfig: ClassesConfig? = classesConfig.classes[index]
      (subLevel == null && subClassesConfig == null) ||
        (subLevel != null && subClassesConfig != null && subLevel.isCompatible(subClassesConfig))
    }
  }
}
