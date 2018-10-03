/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.hanclassifier

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * A sentence as list of encoded tokens.
 *
 * @property tokens the list of tokens encodings
 */
data class EncodedSentence(val tokens: List<DenseNDArray>)
