/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.hanclassifier.dataset

/**
 * An example to train or test the HAN classifier.
 *
 * @property inputText the input text as list of sentences (list of tokens themselves)
 * @property outputGold the index of the gold class
 */
data class Example(val inputText: List<List<String>>, val outputGold: Int)
