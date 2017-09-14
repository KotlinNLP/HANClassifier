/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.hanclassifier.utils

import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.han.HierarchyGroup
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.han.HierarchySequence
import com.kotlinnlp.simplednn.deeplearning.embeddings.EmbeddingsContainer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * Map the forms of the tokens with an unique integer number.
 */
private val formsMap = mutableMapOf<String, Int>()

/**
 *
 */
fun List<List<String>>.toHierarchyGroup(embeddings: EmbeddingsContainer): HierarchyGroup {

  return HierarchyGroup(*Array(
    size = this.size,
    init = { i -> this[i].toHierarchySequence(embeddings) }
  ))
}

/**
 *
 */
private fun List<String>.toHierarchySequence(embeddings: EmbeddingsContainer): HierarchySequence {

  return HierarchySequence(*Array(
    size = this.size,
    init = { i -> getEmbeddingsArrayByForm(form = this[i], embeddings = embeddings) }
  ))
}

/**
 *
 */
private fun getEmbeddingsArrayByForm(form: String, embeddings: EmbeddingsContainer): DenseNDArray {

  val index: Int = getIndexByForm(form)

  return if (index in 0 until embeddings.count)
    embeddings.getEmbeddingByInt(index).array.values
  else
    embeddings.unknownEmbedding.array.values
}

/**
 *
 */
private fun getIndexByForm(form: String): Int {

  if (!formsMap.containsKey(form)) {
    formsMap[form] = formsMap.size
  }

  return formsMap[form]!!
}
