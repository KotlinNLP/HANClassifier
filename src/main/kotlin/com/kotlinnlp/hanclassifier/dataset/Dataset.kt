/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.hanclassifier.dataset

import com.kotlinnlp.hanclassifier.ClassesConfig

/**
 * A dataset to train and test a HAN classifier.
 *
 * @property training the training set as list of examples
 * @property validation the validation set as list of examples
 * @property test the test set as list of examples
 * @param autoComplete whether to autocomplete the classes configuration of the training set in case the examples do not
 *                     cover all the possible classes
 */
class Dataset(
  val training: List<Example>,
  val validation: List<Example>,
  val test: List<Example>,
  autoComplete: Boolean = false
) {

  /**
   * A mutable classes configuration.
   *
   * @property levels the classes indices associated to the configuration of their sub-levels
   */
  private data class MutableClassesConfig(val levels: MutableMap<Int, MutableClassesConfig> = mutableMapOf()) {

    /**
     * Convert this mutable configuration to a classes configuration.
     *
     * @param autoComplete whether to autocomplete the classes configuration in case of missing levels
     *
     * @return a classes configuration
     */
    fun toClassesConfig(autoComplete: Boolean): ClassesConfig? = if (this.levels.isNotEmpty())
      ClassesConfig(
        classes = if (autoComplete)
          (0 .. this.levels.keys.max()!!).associate { it to this.levels[it]?.toClassesConfig(autoComplete = true) }
        else
          this.levels.mapValues { it.value.toClassesConfig(autoComplete = false) })
    else
      null
  }

  /**
   * The hierarchical configuration of the classes defined in the dataset.
   */
  val classesConfig: ClassesConfig = this.getClassesConfig(this.training, autoComplete = autoComplete)

  /**
   * Check the validity and the compatibility of the dataset.
   */
  init {

    require(this.classesConfig.isComplete()) { "The training dataset must contain all the possible classes." }

    require(this.getClassesConfig(this.validation, autoComplete = false).isCompatible(this.classesConfig)) {
      "The classes defined in the validation dataset must be compatible with the training set."
    }

    require(this.getClassesConfig(this.test, autoComplete = false).isCompatible(this.classesConfig)) {
      "The classes defined in the test dataset must be compatible with the training set."
    }
  }

  /**
   * @param examples a list of examples
   * @param autoComplete whether to autocomplete the classes configuration in case the examples do not cover all the
   *                     possible classes
   *
   * @return the sets of classes of the given examples, in a hierarchical configuration
   */
  private fun getClassesConfig(examples: List<Example>, autoComplete: Boolean): ClassesConfig {

    val mutableConfig = MutableClassesConfig()

    examples.forEach { example ->

      var curConfig: MutableClassesConfig = mutableConfig

      example.goldClasses.forEach { curConfig = curConfig.levels.getOrPut(it) { MutableClassesConfig() } }
    }

    return mutableConfig.toClassesConfig(autoComplete)!!
  }
}
