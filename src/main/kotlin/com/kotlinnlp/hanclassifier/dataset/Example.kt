/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.hanclassifier.dataset

import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken

/**
 * An example to train or test the HAN classifier.
 *
 * @property sentences a list of sentences
 * @property goldClasses the indices of the gold classes, in hierarchical order from the top to the bottom
 */
data class Example(val sentences: List<Sentence<FormToken>>, val goldClasses: List<Int>)
