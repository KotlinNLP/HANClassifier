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
 */
data class Dataset(val training: List<Example>, val validation: List<Example>, val test: List<Example>) {

  /**
   * A mutable classes configuration.
   *
   * @property levels the classes indices associated to the configuration of their sub-levels
   */
  private data class MutableClassesConfig(val levels: MutableMap<Int, MutableClassesConfig> = mutableMapOf()) {

    /**
     * Convert this mutable configuration to a classes configuration.
     *
     * @return a classes configuration
     */
    fun toClassesConfig(): ClassesConfig? = if (this.levels.isNotEmpty())
      ClassesConfig(classes = this.levels.mapValues { it.value.toClassesConfig() })
    else
      null
  }

  /**
   * The hierarchical configuration of the classes defined in the dataset.
   */
  val classesConfig: ClassesConfig = this.getClassesConfig(this.training)

  /**
   * Check the validity and the compatibility of the dataset.
   */
  init {

    require(this.classesConfig.isComplete()) { "The training dataset must contain all the possible classes." }

    require(this.getClassesConfig(this.validation).isCompatible(this.classesConfig)) {
      "The classes defined in the validation dataset must be compatible with the training set."
    }

    require(this.getClassesConfig(this.test).isCompatible(this.classesConfig)) {
      "The classes defined in the test dataset must be compatible with the training set."
    }
  }

  /**
   * @param examples a list of examples
   *
   * @return the sets of classes of the given examples, in a hierarchical configuration
   */
  private fun getClassesConfig(examples: List<Example>): ClassesConfig {

    val mutableConfig = MutableClassesConfig()

    examples.forEach { example ->

      var curConfig: MutableClassesConfig = mutableConfig

      example.goldClasses.forEach { curConfig = curConfig.levels.getOrPut(it) { MutableClassesConfig() } }
    }

    return mutableConfig.toClassesConfig()!!
  }
}
