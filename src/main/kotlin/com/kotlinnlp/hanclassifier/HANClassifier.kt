/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.hanclassifier

import com.kotlinnlp.hanclassifier.utils.toHierarchyGroup
import com.kotlinnlp.simplednn.deeplearning.attention.han.HANEncoder
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * A classifier based on Hierarchic Attention Networks.
 *
 * @param model the model of this [HANClassifier]
 */
class HANClassifier(val model: HANClassifierModel) {

  /**
   * The [HANEncoder] used as classifier (Softmax output activation).
   */
  val encoder = HANEncoder<DenseNDArray>(model = this.model.han)

  /**
   * Classify the given [text].
   *
   * @param text a tokenized text as list of sentences (lists of tokens)
   *
   * @return the probability distribution of the classification
   */
  fun classify(text: List<List<String>>): DenseNDArray {
    return this.encoder.forward(text.toHierarchyGroup(this.model.embeddings))
  }
}
