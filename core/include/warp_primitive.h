#pragma once

// return the position of the most significant bit of bit_mask
__device__ __forceinline__ unsigned find_ms_bit(uint bit_mask) {
	uint ret_val;
	asm volatile (
			"bfind.u32 %0, %1;"
			: "=r" (ret_val) : "r"(bit_mask)
	);
	return ret_val;
}

// return the number of one bits in bit_mask
__device__ __forceinline__ unsigned count_bit(uint bit_mask) {
	uint ret_val;
	asm volatile (
			"popc.b32 %0, %1;"
			: "=r" (ret_val) : "r"(bit_mask)
	);
	return ret_val;
}

// find the n-th bit of bit_mask, starting from base, direction with offset
__device__ __forceinline__ unsigned find_nth_bit(uint bit_mask, uint base,
		uint offset) {
	uint ret_val;
	asm volatile (
			"fns.b32 %0, %1, %2, %3;"
			: "=r" (ret_val) :"r" (bit_mask), "r"(base), "r" (offset)
	);
	return ret_val;
}

// set bits from start_pos
__device__ __forceinline__ unsigned set_bits(uint bit_mask, uint val,
		uint start_pos, uint len) {
	uint ret_val;
	asm volatile (
			"bfi.b32 %0, %1, %2, %3, %4;"
			: "=r" (ret_val) :"r" (val), "r"(bit_mask), "r" (start_pos), "r"(len)
	);
	return ret_val;
}

__device__ __forceinline__ unsigned long long extract_bits_64(unsigned long long bit_mask, uint start_pos,
		uint len) {
	unsigned long long ret_val;
	asm volatile (
			"bfe.u64 %0, %1, %2, %3;"
			: "=l" (ret_val) : "l"(bit_mask), "r" (start_pos), "r"(len)
	);
	return ret_val;
}

__device__ __forceinline__ unsigned long long set_bits_64(unsigned long long bit_mask, unsigned long long val,
		uint start_pos, uint len) {
	unsigned long long ret_val;
	asm volatile (
			"bfi.b64 %0, %1, %2, %3, %4;"
			: "=l" (ret_val) :"l" (val), "l"(bit_mask), "r" (start_pos), "r"(len)
	);
	return ret_val;
}

// memory copy
template<typename eletype>
__device__ __forceinline__ void coop_mem_cpy(void *dst_ptr, void *src_ptr, int size, int lane_id)
{
	int act_size = size * sizeof(eletype) / sizeof(int);
	int *src = (int*)src_ptr;
	int *dst = (int*)dst_ptr;
	for (int i = lane_id; i < act_size; i+=WARP_SIZE)
	{
		dst[i] = src[i];
	}
}