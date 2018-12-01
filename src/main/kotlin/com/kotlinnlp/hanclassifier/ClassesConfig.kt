/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.hanclassifier

import java.io.Serializable

/**
 * The configuration of a hierarchy of classes for the classification.
 *
 * @property classes a map of classes indices to the configuration (optional) of their sub-levels
 */
data class ClassesConfig(val classes: Map<Int, ClassesConfig?>) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * The depth of the hierarchy.
   */
  val depth: Int by lazy {
    this.classes.values
      .asSequence()
      .map { 1 + (it?.depth ?: 0) }
      .max()!!
  }

  /**
   * A classes configuration is considered complete if it defines a complete classes set, sequential and starting
   * from 1.
   *
   * @return true if this classes configuration is complete, otherwise false
   */
  fun isComplete(): Boolean =
    this.classes.keys.min() == 0 &&
      this.classes.keys.max() == (this.classes.size - 1) &&
      this.classes.values.all { it?.isComplete() ?: true }

  /**
   * @param other another classes configuration
   *
   * @return true if all this configuration classes and their sub-classes are defined in the other, otherwise false
   */
  fun isCompatible(other: ClassesConfig): Boolean =

    this.classes.all { (classIndex, subLevel) ->

      val otherSubLevel: ClassesConfig? = other.classes[classIndex]

      when {
        classIndex !in other.classes -> false
        subLevel == null -> otherSubLevel == null
        else -> otherSubLevel != null && subLevel.isCompatible(otherSubLevel)
      }
    }
}
