/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.hanclassifier.dataset

import com.kotlinnlp.hanclassifier.EncodedSentence

/**
 * An example to train or test the HAN classifier.
 *
 * @property encodedSentences a list of encoded sentences
 * @property outputGold the index of the gold class
 */
data class Example(val encodedSentences: List<EncodedSentence>, val outputGold: Int)
