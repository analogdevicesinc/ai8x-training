/*
 * On-device execution
 */

#include <arm_math.h>
#include <arm_nnfunctions.h>

#define CNN_START LED_On(0)
#define CNN_COMPLETE LED_Off(0)

arm_status
arm_fully_connected_q7_q8p7_opt(const q7_t * pV,
                                const q7_t * pM,
                                const uint16_t dim_vec,
                                const uint16_t num_of_rows,
                                const uint16_t bias_shift,
                                const uint16_t out_shift, const q7_t * bias, q15_t * pOut, q15_t * vec_buffer);


void arm_softmax_q8p7_q15(const q15_t * vec_in, const uint16_t dim_vec, q15_t * p_out);

void arm_softmax_q8p7_q15_frac(const q15_t * vec_in, const uint16_t dim_vec, q15_t * p_out);

