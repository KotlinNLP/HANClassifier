/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.hanclassifier.helpers

import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.tokensencoder.TokensEncoder
import com.kotlinnlp.tokensencoder.TokensEncoderModel
import com.kotlinnlp.utils.ItemsPool

/**
 * A pool of [TokensEncoder].
 *
 * @param model the model of the tokens encoder
 * @param useDropout whether to apply the dropout of the input
 */
internal class TokensEncodersPool(
  private val model: TokensEncoderModel<FormToken, Sentence<FormToken>>,
  private val useDropout: Boolean
) : ItemsPool<TokensEncoder<FormToken, Sentence<FormToken>>>() {

  /**
   * The factory of a new item.
   *
   * @param id the unique id of the item to create
   *
   * @return a new item with the given [id]
   */
  override fun itemFactory(id: Int): TokensEncoder<FormToken, Sentence<FormToken>> =
    this.model.buildEncoder(useDropout = this.useDropout, id = id)

  /**
   * Release all the items of the pool and return a given number of available encoders.
   *
   * @param size the number of tokens encoder to return
   *
   * @return a list of tokens encoders
   */
  fun getEncoders(size: Int): List<TokensEncoder<FormToken, Sentence<FormToken>>> {

    this.releaseAll()

    return List(size = size, init = { this.getItem() })
  }
}
