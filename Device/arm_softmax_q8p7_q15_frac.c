/*
 * Copyright (C) 2010-2018 Arm Limited or its affiliates. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/* 
 * Portions Copyright (C) 2019 Maxim Integrated Products, Inc.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_softmax_q8p7_q15_frac.c
 * Description:  Q8.7 softmax function with Q15 output
 *
 * $Date:        20. February 2018
 * $Revision:    V.1.0.0
 *
 * Target Processor:  Cortex-M cores
 *
 * -------------------------------------------------------------------- */

#include "arm_math.h"
#include "arm_nnfunctions.h"

/**
 *  @ingroup groupNN
 */

/**
 * @addtogroup Softmax
 * @{
 */

  /**
   * @brief Q8.7 fixed point softmax function, returns Q15
   * @param[in]       vec_in      pointer to input vector
   * @param[in]       dim_vec     input vector dimention
   * @param[out]      p_out       pointer to output vector
   * @return none.
   *
   * @details
   *
   *  Here, instead of typical e based softmax, we use
   *  2-based softmax, i.e.,:
   *
   *  y_i = 2^(x_i/128) / sum(2^(x_j/128))
   *
   *  The relative output will be different here.
   *  But mathematically, the gradient will be the same
   *  with a log(2) scaling factor.
   */

void arm_softmax_q8p7_q15_frac(const q15_t * vec_in, const uint16_t dim_vec, q15_t * p_out)
{
    q31_t     sum, f, r, s, shift;
    int16_t   i;
    q31_t     base;
    base = -1 * 0x80000000; /* int_min = -2^31 */

    /* Find max(vec_in[]) */
    for (i = 0; i < dim_vec; i++)
    {
        if (vec_in[i] > base)
        {
            base = vec_in[i];
        }
    }

    /* we ignore really small values  
     * anyway, they will be 0 after shrinking
     * to q15_t
     */
#define BASE_OFFS 2048
    base = base - BASE_OFFS;

    sum = 0;

    for (i = 0; i < dim_vec; i++)
    {
        if (vec_in[i] > base)
        {
            /* Calculate 2^(vec_in[i] - base) and add to sum */
            shift = (uint32_t)__USAT(vec_in[i] - base, 12);
            /* Separate integer and fractional parts */
            s = (shift + 64) & ~0x7f; /* Round to nearest integer */
            f = (shift - s) << (16 - 7); /* [-0.5, 0.5] Use all of the space we have */
            s = ((15 << 16) - (s << (16 - 7))) >> 16;
            /* Approximate 2^f in [-0.5, 0.5] */
            r = 0x00000e20;                 /* 0.0551716691 * 2^16  */
            r = (r * f + 0x3e1cc333) >> 17; /* 0.2426111222 * 2^32 + 2^16 */
            r = (r * f + 0x58bd46a6) >> 16; /* 0.6932609855 * 2^31 + 2^15 */
            r = r * f + 0x7ffde4a3;         /* 0.9999280735 * 2^30 + 2^14 */
            sum += (uint32_t) r >> (s + 16);
        }
    }

    /* This is effectively (0x1 << 32) / sum */
    int64_t div_base = 0x100000000LL;
    int32_t output_base = (int32_t)(div_base / sum);

    /* Final confidence will be output_base >> ( 17 - (vec_in[i] - base) )
     * so 32768 (0x1<<15) -> 100% confidence when sum = 0x1 << 16, output_base = 0x1 << 16
     * and vec_in[i]-base = 16
     */
    for (i = 0; i < dim_vec; i++)
    {
        if (vec_in[i] > base) 
        {
            /* Here minimum value of 17+base-vec[i] will be 1 */
            /* Calculate 2*-(vec_in[i] - min(vec_in[])) and divide by the sum(vec_in[]) */
            shift = (uint32_t)__USAT(BASE_OFFS + 1 + base - vec_in[i], 12);

            /* Separate integer and fractional parts */
            s = (shift + 64) & ~0x7f; /* Round to nearest integer */
            f = (s - shift) << (16 - 7); /* [-0.5, 0.5] Use all of the space we have */
            s = ((15 << 16) - (s << (16 - 7))) >> 16;
            /* Approximate 2^f in [-0.5, 0.5] */
            r = 0x00000e20;                 /* 0.0551716691 * 2^16  */
            r = (r * f + 0x3e1cc333) >> 17; /* 0.2426111222 * 2^32 + 2^16 */
            r = (r * f + 0x58bd46a6) >> 16; /* 0.6932609855 * 2^31 + 2^15 */
            r = r * f + 0x7ffde4a3;         /* 0.9999280735 * 2^30 + 2^14 */
            s = (uint32_t) r >> (31 - s);   /* = 2^15 * 2^-shift */
            p_out[i] = (q15_t) (output_base * s >> 16);

            // printf("shift=%d, f=%d, r=%08x, s=%08x, s/2^15=%f, x/shift/128=%f, 2^x=%f * %08x -> %08x\n",
            //        -shift, f, r, s, (float) s / 32768.0, (float) shift / -128.0,
            //        exp2f((float) shift / -128.0), (q15_t) output_base, p_out[i]);
        } else
        {
            p_out[i] = 0;
        }
    }

}

/**
 * @} end of Softmax group
 */
