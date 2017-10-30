/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.hanclassifier.utils

import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.han.HierarchyGroup
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.han.HierarchySequence
import com.kotlinnlp.simplednn.deeplearning.embeddings.EmbeddingsContainerByStrings
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 *
 */
fun List<List<String>>.toHierarchyGroup(embeddings: EmbeddingsContainerByStrings): HierarchyGroup {

  return HierarchyGroup(*Array(
    size = this.size,
    init = { i -> this[i].toHierarchySequence(embeddings) }
  ))
}

/**
 *
 */
private fun List<String>.toHierarchySequence(embeddings: EmbeddingsContainerByStrings): HierarchySequence<DenseNDArray> {

  return HierarchySequence(*Array(
    size = this.size,
    init = { i -> embeddings.getEmbedding(id = this[i]).array.values }
  ))
}
