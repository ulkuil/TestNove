#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "xmmintrin.h"


#define stdcall __attribute__((regparm(3)))


#define SSELoad(align, S, B) B = align ? \
	_mm_load_si128(S) : _mm_loadu_si128(S);

#define MoveLoad(align, S, B) align ? \
	_mm_store_si128(S, B) : _mm_storeu_si128(S, B);

#define MoveSSECount(Count, d, s, align, o) \
	__m128i xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;\
	switch(Count) { \
		case 8: SSELoad(align, (__m128i*)(s+o) + 7, xmm8); \
		case 7: SSELoad(align, (__m128i*)(s+o) + 6, xmm7); \
		case 6: SSELoad(align, (__m128i*)(s+o) + 5, xmm6); \
		case 5: SSELoad(align, (__m128i*)(s+o) + 4, xmm5); \
		case 4: SSELoad(align, (__m128i*)(s+o) + 3, xmm4); \
		case 3: SSELoad(align, (__m128i*)(s+o) + 2, xmm3); \
		case 2: SSELoad(align, (__m128i*)(s+o) + 1, xmm2); \
		case 1: SSELoad(align, (__m128i*)(s+o), xmm1); \
		case 0: break; \
	} \
	switch(Count) { \
		case 8: MoveLoad(align, (__m128i*)(d+o) + 7, xmm8); \
		case 7: MoveLoad(align, (__m128i*)(d+o) + 6, xmm7); \
		case 6: MoveLoad(align, (__m128i*)(d+o) + 5, xmm6); \
		case 5: MoveLoad(align, (__m128i*)(d+o) + 4, xmm5); \
		case 4: MoveLoad(align, (__m128i*)(d+o) + 3, xmm4); \
		case 3: MoveLoad(align, (__m128i*)(d+o) + 2, xmm3); \
		case 2: MoveLoad(align, (__m128i*)(d+o) + 1, xmm2); \
		case 1: MoveLoad(align, (__m128i*)(d+o), xmm1); \
		case 0: break; \
	}

#define SetType(t, s, d, o) *((t*)(s + o)) = *((t*)(d + o))

#define Mov1B(d, s, o) SetType(uint8_t, s, d, o)
#define Mov2B(d, s, o) SetType(uint16_t, s, d, o)
#define Mov4B(d, s, o) SetType(uint32_t, s, d, o)
#define Mov8B(d, s, o) SetType(uint64_t, s, d, o)

#define Mov3B(d, s, o) Mov1B(d, s, o); Mov2B(d, s, o + 1); 
#define Mov5B(d, s, o) Mov4B(d, s, o); Mov1B(d, s, o + 4);
#define Mov6B(d, s, o) Mov4B(d, s, o); Mov2B(d, s, o + 4);
#define Mov7B(d, s, o) Mov4B(d, s, o); Mov4B(d, s, o + 4 + 4 - 1);
#define Mov9B(d, s, o)  Mov8B(d, s, o); Mov1B(d, s, o +8);
#define Mov10B(d, s, o) Mov8B(d, s, o); Mov2B(d, s, o +8);
#define Mov11B(d, s, o) Mov8B(d, s, o); Mov4B(d, s, o + 8 - 1);
#define Mov12B(d, s, o) Mov8B(d, s, o); Mov4B(d, s, o + 8);
#define Mov13B(d, s, o) Mov8B(d, s, o); Mov5B(d, s, o + 8);
#define Mov14B(d, s, o) Mov8B(d, s, o); Mov6B(d, s, o + 8);
#define Mov15B(d, s, o) Mov8B(d, s, o); Mov8B(d, s, o + 8);


#define MoveSSE(i, r, align, dd, ss) \
	case i+1: \
		{ MoveSSECount(r, dd, ss, align, 0); } \
		Mov1B(dd, ss,  16 * r); \
	break; \
	case i+2: \
		{ MoveSSECount(r, dd, ss, align, 0); } \
		Mov2B(dd, ss, 16 * r); \
	break; \
	case i+3: \
		{ MoveSSECount(r, dd, ss, align, 0); } \
		 Mov4B(dd, ss,  ((16 * r) - 1)); \
	break; \
	case i+4: \
		{ MoveSSECount(r, dd, ss, align, 0); } \
		Mov4B(dd, ss, 16 * r); \
	break; \
	case i+5: \
		{ MoveSSECount(r, dd, ss, align, 0); } \
		Mov5B(dd, ss, 16 * r); \
	break; \
	case i+6: \
		{ MoveSSECount(r, dd, ss, align, 0); } \
		Mov6B(dd, ss, 16 * r); \
	break; \
	case i+7: \
		{ MoveSSECount(r, dd, ss, align, 0); } \
		Mov7B(dd, ss, 16 * r); \
	break; \
	case i+8: \
		{ MoveSSECount(r, dd, ss, align, 0); } \
		Mov8B(dd, ss, 16 * r); \
	break; \
	case i+9: \
		{ MoveSSECount(r, dd, ss, align, 0); } \
		Mov9B(dd, ss, 16 * r); \
	break; \
	case i+10: \
		{ MoveSSECount(r, dd, ss, align, 0); } \
		Mov10B(dd, ss, 16 * r); \
	break; \
	case i+11: \
		{ MoveSSECount(r, dd, ss, align, 0); } \
		Mov11B(dd, ss, 16 * r); \
	break; \
	case i+12: \
		{ MoveSSECount(r, dd, ss, align, 0); } \
		Mov12B(dd, ss, 16 * r); \
	break; \
	case i+13: \
		{ MoveSSECount(r, dd, ss, align, 0); } \
		Mov13B(dd, ss, 16 * r); \
	break; \
	case i+14: \
		{ MoveSSECount(r, dd, ss, align, 0); } \
		Mov14B(dd, ss, 16 * r); \
	break; \
	case i+15: \
		{ MoveSSECount(r, dd, ss, align, 0); } \
		Mov15B(dd, ss, 16 * r); \
	break; \
	case i+16: \
		{ MoveSSECount(r+1, dd, ss, align, 0); } \
	break;


#define MoveLen(dd, ss, len, Align) \
		switch(len){ \
			case 0: return; \
			MoveSSE(16 * 0 + 1, 0, Align, dd, ss); \
			MoveSSE(16 * 1 + 1, 1, Align, dd, ss); \
			MoveSSE(16 * 2 + 1, 2, Align, dd, ss); \
			MoveSSE(16 * 3 + 1, 3, Align, dd, ss); \
			MoveSSE(16 * 4 + 1, 4, Align, dd, ss); \
			MoveSSE(16 * 5 + 1, 5, Align, dd, ss); \
			MoveSSE(16 * 6 + 1, 6, Align, dd, ss); \
			MoveSSE(16 * 7 + 1, 7, Align, dd, ss); \
		default: \
			while(len > 128) { \
				MoveSSECount(8, dd, ss, Align, 0); \
				len -= 128; \
				dd -= 128; \
				ss -= 128; \
			} \
		}

stdcall void MoveMemory(const char* Source, char* dest,
 size_t len) {

	int AlignSS = (int)Source & 15;
	if(AlignSS == ((int)dest & 15)) {
		if(AlignSS) {
			AlignSS = 16 - AlignSS;
			switch(AlignSS)  {
				MoveSSE(16 * 0 + 1, 0, 1, dest, Source);
			}
			dest += AlignSS;
			Source += AlignSS;
			len -= AlignSS;
		}
		MoveLen(dest, Source, len, 1);
	} else {
		MoveLen(dest, Source, len, 0);
	}
}

